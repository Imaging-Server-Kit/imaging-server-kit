from typing import Optional

import numpy as np

from imaging_server_kit.merge.layer_merger import Merger
from imaging_server_kit.types._mask import Mask
import networkx as nx
from skimage.util import map_array


class MaskOverrideMerger(Merger):
    """Merge masks using and `override` strategy: last tile overrides existing data in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Mask, incoming_layer: Mask) -> None:
        if (incoming_layer.data is None) or (incoming_layer.coords_max is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.coords_max is None):
            receiving_layer.data = receiving_layer.initialize([1] * incoming_layer.ndim)
            receiving_layer.meta = incoming_layer.meta

        if (receiving_layer.coords_max is None) or (receiving_layer.data is None):
            return  # This should never happen (just there for type hints)

        # Simple "Override" strategy; could be improved with pixel-wise majority voting between overlapping tiles
        _slices = incoming_layer.domain.slices
        _stack = np.stack([receiving_layer.coords_max, incoming_layer.coords_max])
        _bounds = np.max(_stack, axis=0).tolist()

        if _bounds != receiving_layer.coords_max:
            new_data = incoming_layer.initialize(_bounds)
            new_data[receiving_layer.domain.slices] = receiving_layer.data
        else:
            new_data = receiving_layer.data

        # `Override` strategy:
        new_data[_slices] = incoming_layer.data
        receiving_layer.data = new_data
        receiving_layer.meta = incoming_layer.meta

    @staticmethod
    def on_first_merge(receiving_layer: Mask, incoming_layer: Mask):
        receiving_layer.meta = incoming_layer.meta

    @staticmethod
    def on_last_merge(receiving_layer: Mask, incoming_layer: Mask):
        pass


class MaskTileOverrideMerger(MaskOverrideMerger):
    """Merge two masks using an `override` strategy: incoming data overrides existnig data."""

    def on_first_merge(self, receiving_layer: Mask, incoming_layer: Mask):
        receiving_layer.data = incoming_layer.initialize(receiving_layer.coords_max)
        receiving_layer.meta = incoming_layer.meta


class InstanceTileTracker:
    def __init__(self) -> None:
        self.initialize()

    def initialize(self):
        self.N = 0  # Current number of objects
        self.G = nx.Graph()

    def add_N_to_tile(self, labels: np.ndarray) -> np.ndarray:
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

    def resolve(self, arr):
        if not hasattr(self, "_mapping"):
            self.build_mapping()
        input_vals = np.array(list(self._mapping.keys()), dtype=np.int64)
        output_vals = np.array(list(self._mapping.values()), dtype=np.int64)
        return map_array(arr, out=arr, input_vals=input_vals, output_vals=output_vals)


def unique_positive(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels[labels > 0])


def overlap_border_mask(layer: Mask) -> Optional[np.ndarray]:
    """Returns a boolean array selecting the rectangular region overalpping with other tiles."""
    if (layer.tile_meta.overlap_px is None) or (layer.size is None):
        return

    overlap_slices = tuple(
        [
            slice(pos, max_pos - pos)
            for pos, max_pos in zip(layer.tile_meta.overlap_px, layer.size)
        ]
    )
    mask = np.ones(layer.size)
    mask[overlap_slices] = 0
    return mask == 1


class InstanceMaskTileMerger(Merger):
    def __init__(self, min_intersecting_px: int = 1) -> None:
        self.min_intersecting_px = min_intersecting_px
        self.tile_tracker = InstanceTileTracker()

    def merge(self, receiving_layer: Mask, incoming_layer: Mask) -> None:
        if (incoming_layer.data is None) or (incoming_layer.coords_max is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.coords_max is None):
            receiving_layer.data = incoming_layer.initialize(
                tuple([1] * incoming_layer.ndim)
            )
            receiving_layer.meta = incoming_layer.meta

        if (receiving_layer.coords_max is None) or (receiving_layer.data is None):
            return  # This should never happen (just there for type hints)

        _slices = incoming_layer.domain.slices
        _stack = np.stack([receiving_layer.coords_max, incoming_layer.coords_max])
        _bounds = np.max(_stack, axis=0).tolist()

        if _bounds != receiving_layer.coords_max:
            new_data = incoming_layer.initialize(_bounds)
            new_data[receiving_layer.domain.slices] = receiving_layer.data
        else:
            new_data = receiving_layer.data

        receiving_layer.data = new_data  # Extend the source layer data

        src_tile = receiving_layer.select(domain=incoming_layer.domain)
        if src_tile.data is None:
            raise ValueError(f"Could not get a mask tile where it was requested.")

        dst_arr = self.tile_tracker.add_N_to_tile(incoming_layer.data)

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

        new_data[_slices] = dst_arr
        receiving_layer.data = new_data
        receiving_layer.meta = incoming_layer.meta

    def on_first_merge(self, receiving_layer: Mask, incoming_layer: Mask):
        receiving_layer.data = incoming_layer.initialize([1] * incoming_layer.ndim)
        self.tile_tracker = InstanceTileTracker()

    def on_last_merge(self, receiving_layer: Mask, incoming_layer: Mask):
        receiving_layer.data = self.tile_tracker.resolve(receiving_layer.data)
        self.tile_tracker = InstanceTileTracker()
