import warnings
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import networkx as nx
from skimage.util import map_array

from imaging_server_kit.merge.merger import DefaultMerger
from imaging_server_kit.types._mask import Mask
from imaging_server_kit.merge.common import prepare_canvas, update_meta
from imaging_server_kit.core.tiling import generate_tiles

# Max pixels for doing the resolve() operation of instance masks one go (set arbitrarily, could be configurable in future versions).
# If the image is bigger than that, we do the instance mask resolution in tiles, too.
MAX_MAP_ARRAY_SIZE = 1024**3


class MaskOverrideMerger(DefaultMerger):
    """Merge masks using and `override` strategy: last tile overrides existing data in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Mask, incoming_layer: Mask) -> None:
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.position is None):
            receiving_layer.position = incoming_layer.position
            receiving_layer.data = incoming_layer.data
            receiving_layer.meta = incoming_layer.meta
            return

        new_data, slices_with_channel = prepare_canvas(
            receiving_layer, incoming_layer, dtype=np.uint32, copy_data=False
        )

        # Simply override the data
        new_data[slices_with_channel] = incoming_layer.data

        # Update the data of receiving layer
        receiving_layer.data = new_data

        # Meta becomes incoming layer's meta (except from position)
        update_meta(receiving_layer, incoming_layer)


class InstanceTileTracker:
    def __init__(self) -> None:
        self.initialize()

    def initialize(self):
        self.N = 0  # Current number of objects
        self.G = nx.Graph()

        # Used to store features during the tiles run => resolved at the end with self.resolve_features()
        self.F = {}

    def add_N_to_tile(
        self, labels: np.ndarray, features: Optional[Dict] = None
    ) -> np.ndarray:
        # Important - copy to avoid mutating the original array. We use uint32 so that
        # offsetting the labels by N cannot overflow the input dtype (e.g. uint16).
        labels = labels.astype(np.uint32)
        
        if labels.sum() > 0:
            labels[labels != 0] = labels[labels != 0] + self.N

            if features:
                for feature_key, feature_val in features.items():
                    if feature_key == "label":
                        is_bg = feature_val == 0
                        # Make the labels global (except background)
                        feature_val = feature_val + self.N
                        feature_val[is_bg] = 0

                    # Store a list of feature arrays as self.F
                    if feature_key not in self.F:
                        self.F[feature_key] = [feature_val]
                    else:
                        self.F[feature_key].append(feature_val)

            self.N = labels.max()

        return labels

    def add_node(self, lab):
        self.G.add_node(lab)

    def add_edge(self, a, b):
        self.G.add_edge(a, b)

    def build_mapping(self):
        self.G.add_nodes_from(range(1, self.N + 1))
        mapping = {}
        for comp_id, comp in enumerate(nx.connected_components(self.G), start=1):
            for n in comp:
                mapping[int(n)] = comp_id
        self._mapping = mapping

    def resolve(self, arr: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_mapping"):
            self.build_mapping()

        input_vals = np.array(list(self._mapping.keys()), dtype=np.int64)
        output_vals = np.array(list(self._mapping.values()), dtype=np.int64)

        arr = np.ascontiguousarray(arr)

        return map_array(arr, out=arr, input_vals=input_vals, output_vals=output_vals)

    def resolve_features(self, old_unique_labels: np.ndarray) -> Dict:
        if not hasattr(self, "_mapping"):
            self.build_mapping()

        input_vals = np.array(list(self._mapping.keys()), dtype=np.int64)
        output_vals = np.array(list(self._mapping.values()), dtype=np.int64)

        resolved_features = {}
        for feature_key, feature_val in self.F.items():
            # feature_val should be a list of 1D arrays of different lengths => we hstack them
            resolved_features[feature_key] = np.hstack(feature_val)

        try:
            # Sort values based on the `label` in ascending order
            df = pd.DataFrame(resolved_features).sort_values(by="label", ascending=True)
        except ValueError as e:
            feature_lengths = [len(v) for v in resolved_features.values()]
            warnings.warn(
                f"Features could not be resolved into a DataFrame. Perhaps feature values don't have the same lengths? (Lengths: {feature_lengths}). Error: {e}",
                stacklevel=2,
            )
            return {}

        # Remove rows where `label` is not in `old_unique_labels` (labels present in the array to resolve)
        # Note: slightly unclear where this difference comes from..
        df = df[df["label"].isin(old_unique_labels)]

        # Remap the labels
        df["label"] = map_array(
            df["label"].values.copy(), input_vals=input_vals, output_vals=output_vals
        )

        # Feature aggregation:
        # For all 'labels' that have duplicates (= they fall on a tile boundary)
        # we drop the duplicate rows, keep the `label` since it is correct, and
        # for all other features we set the values to NaN. This is because, there
        # is no good default way to know how to aggregate arbitrary features at the
        # tile boundary. In the future, we could specify strategies for specific,
        # common features like `area` (which should be added) or `class` (majority voting).
        is_dup_label = df["label"].duplicated(keep=False)
        df = df.drop_duplicates(subset=["label"], keep="first")
        other_cols = df.columns.difference(["label"])
        df.loc[df["label"].isin(df.loc[is_dup_label, "label"]), other_cols] = np.nan

        # Return as dict of 1D numpy arrays
        return {c: df[c].values for c in df.columns}


def unique_positive(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels[labels > 0])


def overlap_border_mask(layer: Mask) -> Optional[np.ndarray]:
    """Returns a boolean array selecting the rectangular region overalpping with other tiles."""

    # If unspecified, overlap defaults to zero
    _overlap_px = layer.tile_meta.overlap_px
    if _overlap_px is None:
        if layer._bounds is not None:
            _overlap_px = tuple([0] * len(layer._bounds))

    if (_overlap_px is None) or (layer.size is None):
        return

    size_int = tuple([math.ceil(v) for v in layer.size])

    overlap_slices = tuple(
        [slice(pos, max_pos - pos) for pos, max_pos in zip(_overlap_px, size_int)]
    )

    mask = np.ones(size_int, dtype=np.uint8)

    mask[overlap_slices] = 0

    return mask == 1


def clean_mask_layer_features(mask: Mask) -> Optional[Dict]:
    if "features" in mask.meta.keys():
        incoming_features = mask.meta["features"]
        if isinstance(incoming_features, Dict):
            if "label" in incoming_features.keys():
                labels_arr = incoming_features["label"]
                if set(labels_arr) == set(np.unique(mask.data)):
                    # Extract clean features (1D arrays of correct length)
                    clean_features = {"label": labels_arr}
                    for feature_key, feature_val in incoming_features.items():
                        if isinstance(feature_val, np.ndarray):
                            if len(feature_val) == len(labels_arr):
                                clean_features[feature_key] = feature_val
                    return clean_features


class InstanceMaskTileMerger(DefaultMerger):
    def __init__(self, min_intersecting_px: int = 1) -> None:
        self.min_intersecting_px = min_intersecting_px
        self.tile_tracker = InstanceTileTracker()

    def merge(self, receiving_layer: Mask, incoming_layer: Mask) -> None:
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.position is None):
            receiving_layer.position = incoming_layer.position
            # Labels are registered in the tracker, so that labels from the next tiles don't collide with them
            receiving_layer.data = self.tile_tracker.add_N_to_tile(
                incoming_layer.data,
                features=clean_mask_layer_features(incoming_layer),
            )
            receiving_layer.meta = incoming_layer.meta
            return

        new_data, slices_with_channel = prepare_canvas(
            receiving_layer, incoming_layer, dtype=np.uint32, copy_data=False
        )

        receiving_layer.data = new_data  # Extend the source layer data

        src_tile = receiving_layer.select(domain=incoming_layer.extent)
        if src_tile.data is None:
            raise ValueError(f"Could not get a mask tile where it was requested.")

        # Handle label features; features dictionary should contain a `label` column to identify objects.
        features = clean_mask_layer_features(incoming_layer)

        dst_arr = self.tile_tracker.add_N_to_tile(
            incoming_layer.data, features=features
        )

        for new_label in unique_positive(dst_arr):
            self.tile_tracker.add_node(new_label)

        border_mask = overlap_border_mask(incoming_layer)

        if border_mask is not None:
            for dst_lab in unique_positive(dst_arr[border_mask]):
                filt = np.logical_and(border_mask, dst_arr == dst_lab)
                src_tile_filt = src_tile.data[filt]
                for src_lab in unique_positive(src_tile_filt):
                    n_intersecting_px = (src_tile_filt == src_lab).sum()
                    if n_intersecting_px > self.min_intersecting_px:
                        self.tile_tracker.add_edge(src_lab, dst_lab)

        new_data[slices_with_channel] = dst_arr

        # Update the data of receiving layer
        receiving_layer.data = new_data

        # Meta becomes incoming layer's meta (except from position/features)
        update_meta(receiving_layer, incoming_layer, exclude=("position", "features"))

    def on_first_merge(self, receiving_layer: Mask, incoming_layer: Mask):
        self.tile_tracker = InstanceTileTracker()

        if (receiving_layer is incoming_layer) and (incoming_layer.data is not None):
            # The layer was just added to the stack (its data is not merged). We register its labels
            # in the tracker, so that labels from the next tiles don't collide with them.
            receiving_layer.data = self.tile_tracker.add_N_to_tile(
                incoming_layer.data,
                features=clean_mask_layer_features(incoming_layer),
            )

    def on_last_merge(self, receiving_layer: Mask, incoming_layer: Mask):
        from imaging_server_kit.merge.layer_merger import LayerMerger

        old_unique_labels = np.unique(receiving_layer.data)

        if receiving_layer.data.size > MAX_MAP_ARRAY_SIZE:
            # Use ndim to approximate tile size
            tile_size = int(np.floor(MAX_MAP_ARRAY_SIZE ** (1 / receiving_layer.ndim)))

            merger = LayerMerger()
            for tile_meta, tile_domain in generate_tiles(
                domain=receiving_layer.extent, tile_size=tile_size
            ):
                roi = receiving_layer.select(tile_domain)
                roi.data = self.tile_tracker.resolve(roi.data)
                merger.merge(receiving_layer, roi)
        else:
            # Don't bother resolving the data in tiles
            receiving_layer.data = self.tile_tracker.resolve(receiving_layer.data)

        # Resolve the label features (if there are any - otherwise, we don't add an empty features dict.)
        if len(self.tile_tracker.F):
            receiving_layer.meta["features"] = self.tile_tracker.resolve_features(
                old_unique_labels=old_unique_labels
            )

        # Re-initialize the tracker
        self.tile_tracker = InstanceTileTracker()
