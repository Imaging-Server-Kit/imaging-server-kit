import warnings
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import networkx as nx
from skimage.util import map_array

from imaging_server_kit.merge.merger import DefaultMerger
from imaging_server_kit.types._mask import Mask, invalidate_label_features
from imaging_server_kit.merge.common import prepare_canvas, update_meta
from imaging_server_kit.core.tiling import generate_tiles

# Max pixels for doing the resolve() operation of instance masks one go (set arbitrarily, could be configurable in future versions).
# If the image is bigger than that, we do the instance mask resolution in tiles, too.
MAX_MAP_ARRAY_SIZE = 1024**3


def _drop_tiled_features(receiving_layer: Mask, incoming_layer: Mask) -> None:
    """Remove the features of a semantic mask merged from several tiles.

    There is no general way to combine features computed per tile (e.g. the area of each class),
    and keeping those of the last tile would be wrong.
    """
    if incoming_layer.tile_meta.n_tiles <= 1:
        return

    if "features" in incoming_layer.meta:
        warnings.warn(
            f"Features of the semantic mask `{incoming_layer.name}` are dropped: they cannot be merged across tiles.",
            stacklevel=2,
        )

    if (receiving_layer.meta is not None) and ("features" in receiving_layer.meta):
        receiving_layer.meta = {
            k: v for k, v in receiving_layer.meta.items() if k != "features"
        }


class MaskOverrideMerger(DefaultMerger):
    """Merge masks using and `override` strategy: last tile overrides existing data in overlapping regions.

    In tiled runs, features are dropped (they cannot be merged across tiles).
    """

    @staticmethod
    def merge(receiving_layer: Mask, incoming_layer: Mask) -> None:
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.position is None):
            receiving_layer.position = incoming_layer.position
            receiving_layer.data = incoming_layer.data
            receiving_layer.meta = incoming_layer.meta
            _drop_tiled_features(receiving_layer, incoming_layer)
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
        _drop_tiled_features(receiving_layer, incoming_layer)

    @staticmethod
    def on_first_merge(receiving_layer: Mask, incoming_layer: Mask) -> None:
        # Also removes the features of the first tile (added to the stack) or of a previous run
        _drop_tiled_features(receiving_layer, incoming_layer)


class InstanceTileTracker:
    def __init__(self) -> None:
        self.initialize()

    def initialize(self):
        self.N = 0  # Current number of objects
        self.G = nx.Graph()

        # Used to store features (one DataFrame per tile) during the tiles run => resolved at the end with self.resolve_features()
        self.F = []

    def add_N_to_tile(
        self, labels: np.ndarray, features: Optional[Dict] = None
    ) -> np.ndarray:
        # Important - copy to avoid mutating the original array. We use uint32 so that
        # offsetting the labels by N cannot overflow the input dtype (e.g. uint16).
        labels = labels.astype(np.uint32)

        # Features are stored even if the tile is empty, so that features of a previous run get replaced
        if features is not None:
            df = pd.DataFrame(features)
            # Only keep the features of objects present in the tile (this drops the background)
            df = df[df["label"].isin(unique_positive(labels))]
            # Make the labels global
            self.F.append(df.assign(label=df["label"] + self.N))

        if labels.sum() > 0:
            labels[labels != 0] = labels[labels != 0] + self.N
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

    def current_features(self) -> Dict:
        """Features registered so far, matching the labels of the merged mask before it is resolved."""
        df = pd.concat(self.F, ignore_index=True)
        return {c: df[c].values for c in df.columns}

    def resolve_features(self, old_unique_labels: np.ndarray) -> Dict:
        if not hasattr(self, "_mapping"):
            self.build_mapping()

        input_vals = np.array(list(self._mapping.keys()), dtype=np.int64)
        output_vals = np.array(list(self._mapping.values()), dtype=np.int64)

        # Tiles missing a feature column get NaN values for it
        df = pd.concat(self.F, ignore_index=True)

        # Remove rows where `label` is not in `old_unique_labels` (labels present in the array to resolve),
        # e.g. objects overridden by the next tiles
        df = df[df["label"].isin(old_unique_labels)].copy()
        if len(df) == 0:
            return {c: df[c].values for c in df.columns}

        # Remap the labels, and sort rows by label
        df["label"] = map_array(
            df["label"].values.copy(), input_vals=input_vals, output_vals=output_vals
        )
        df = df.sort_values(by="label")

        # Feature aggregation:
        # For all 'labels' that have duplicates (= they fall on a tile boundary)
        # we drop the duplicate rows, keep the `label` since it is correct, and
        # for all other features we set the values to NaN. This is because, there
        # is no good default way to know how to aggregate arbitrary features at the
        # tile boundary. In the future, we could specify strategies for specific,
        # common features like `area` (which should be added) or `class` (majority voting).
        dup_labels = df.loc[df["label"].duplicated(), "label"].unique()
        df = df.drop_duplicates(subset=["label"], keep="first")

        # Return as dict of 1D numpy arrays
        return invalidate_label_features(df, dup_labels)


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
    """Features of the mask with a `label` column, keeping only the 1D arrays of the same length (one value per label)."""
    features = mask.meta.get("features")
    if not isinstance(features, Dict) or ("label" not in features):
        return

    labels_arr = np.asarray(features["label"])
    clean_features = {}
    for feature_key, feature_val in features.items():
        feature_val = np.asarray(feature_val)
        if (feature_val.ndim == 1) and (len(feature_val) == len(labels_arr)):
            clean_features[feature_key] = feature_val
        else:
            warnings.warn(
                f"Feature `{feature_key}` is not a 1D array with one value per label; it is dropped.",
                stacklevel=2,
            )
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
            self._update_features(receiving_layer)
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
        self._update_features(receiving_layer)

    def _update_features(self, layer: Mask) -> None:
        """During the run, the layer shows the features of the tiles merged so far (e.g. live in Napari).

        Without this, it would show the features of the first tile, or of a previous run.
        They are resolved with the labels on the last merge.
        """
        if len(self.tile_tracker.F):
            # New dict: the meta may be shared with the incoming layer
            layer.meta = {**layer.meta, "features": self.tile_tracker.current_features()}

    def on_first_merge(self, receiving_layer: Mask, incoming_layer: Mask):
        self.tile_tracker = InstanceTileTracker()

        if receiving_layer.data is not None:
            # We register the labels (and features) of the receiving layer in the tracker, so that labels from
            # the next tiles don't collide with them. Either the layer was just added to the stack (its data
            # is not merged), or it holds objects from a previous run, outside of the reinitialized domain.
            receiving_layer.data = self.tile_tracker.add_N_to_tile(
                receiving_layer.data,
                features=clean_mask_layer_features(receiving_layer),
            )
            self._update_features(receiving_layer)

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
