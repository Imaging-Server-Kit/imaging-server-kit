from typing import Optional

import numpy as np

from imaging_server_kit.merge.layer_merger import DefaultMerger
from imaging_server_kit.types._mask import Mask
import networkx as nx
from skimage.util import map_array


class MaskOverrideMerger(DefaultMerger):
    """Merge masks using and `override` strategy: last tile overrides existing data in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Mask, incoming_layer: Mask) -> None:
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return
        
        channel_axis = incoming_layer.channel_axis
        if channel_axis is not None:
            n_channels = incoming_layer.data.shape[channel_axis] 

        if (receiving_layer.data is None) or (receiving_layer.ndim is None):
            s = incoming_layer.size
            
            if channel_axis is not None:
                s_with_channel = s[:channel_axis] + (n_channels,) + s[channel_axis:]
            else:
                s_with_channel = s
            
            receiving_layer.data = np.zeros(s_with_channel, dtype=np.uint16)
            
            receiving_layer.domain.coords_min = incoming_layer.domain.coords_min

        min_bounds = np.min(
            np.stack([receiving_layer.coords_min, incoming_layer.coords_min]),
            axis=0,
        )

        max_bounds = np.max(
            np.stack([receiving_layer.coords_max, incoming_layer.coords_max]),
            axis=0,
        )

        size = tuple(max_bounds - min_bounds)

        if size != receiving_layer.size:
            slices_rec = []
            slices = []
            cmin_diff = []
            for receiving_cmin, incoming_cmin, incoming_size, receiving_size in zip(
                receiving_layer.coords_min,
                incoming_layer.coords_min,
                incoming_layer.size,
                receiving_layer.size,
            ):
                diff = incoming_cmin - receiving_cmin
                start = 0 if diff < 0 else diff
                stop = incoming_size + start
                slices.append(slice(start, stop))
                start_receiving = -diff if diff < 0 else 0
                stop_receiving = start_receiving + receiving_size 
                slices_rec.append(slice(start_receiving, stop_receiving))
                cmin_diff.append(start_receiving)
            cmin_diff = np.array(cmin_diff)
            slices_rec = tuple(slices_rec)
            slices = tuple(slices)
            
            if channel_axis is not None:
                size_with_channel = size[:channel_axis] + (n_channels,) + size[channel_axis:]
                slices_rec_with_channel = slices_rec[:channel_axis] + (slice(None),) + slices_rec[channel_axis:]
                slices_with_channel = slices[:channel_axis] + (slice(None),) + slices[channel_axis:]
            else:
                size_with_channel = size
                slices_rec_with_channel = slices_rec
                slices_with_channel = slices
            
            new_data = np.zeros(size_with_channel, dtype=np.uint16)
            
            new_data[slices_rec_with_channel] = receiving_layer.data
            
            receiving_layer.domain.coords_min = tuple(np.array(receiving_layer.domain.coords_min) - cmin_diff)
        else:
            new_data = receiving_layer.data

            slices = []
            for receiving_cmin, incoming_cmin, incoming_size in zip(
                receiving_layer.coords_min,
                incoming_layer.coords_min,
                incoming_layer.size,
            ):
                start = incoming_cmin - receiving_cmin
                stop = incoming_size + start
                slices.append(slice(start, stop))
            slices = tuple(slices)
            
            if channel_axis is not None:
                slices_with_channel = slices[:channel_axis] + (slice(None),) + slices[channel_axis:]
            else:
                slices_with_channel = slices

        # Simply override the data
        new_data[slices_with_channel] = incoming_layer.data

        receiving_layer.data = new_data
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


class InstanceMaskTileMerger(DefaultMerger):
    def __init__(self, min_intersecting_px: int = 1) -> None:
        self.min_intersecting_px = min_intersecting_px
        self.tile_tracker = InstanceTileTracker()

    def merge(self, receiving_layer: Mask, incoming_layer: Mask) -> None:
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return

        channel_axis = incoming_layer.channel_axis
        if channel_axis is not None:
            n_channels = incoming_layer.data.shape[channel_axis] 

        if (receiving_layer.data is None) or (receiving_layer.ndim is None):
            s = incoming_layer.size
            
            if channel_axis is not None:
                s_with_channel = s[:channel_axis] + (n_channels,) + s[channel_axis:]
            else:
                s_with_channel = s
            
            receiving_layer.data = np.zeros(s_with_channel, dtype=np.uint16)
            
            receiving_layer.domain.coords_min = incoming_layer.domain.coords_min
        
        min_bounds = np.min(
            np.stack([receiving_layer.coords_min, incoming_layer.coords_min]),
            axis=0,
        )

        max_bounds = np.max(
            np.stack([receiving_layer.coords_max, incoming_layer.coords_max]),
            axis=0,
        )

        size = tuple(max_bounds - min_bounds)

        if size != receiving_layer.size:
            slices_rec = []
            slices = []
            cmin_diff = []
            for receiving_cmin, incoming_cmin, incoming_size, receiving_size in zip(
                receiving_layer.coords_min,
                incoming_layer.coords_min,
                incoming_layer.size,
                receiving_layer.size,
            ):
                diff = incoming_cmin - receiving_cmin
                start = 0 if diff < 0 else diff
                stop = incoming_size + start
                slices.append(slice(start, stop))
                start_receiving = -diff if diff < 0 else 0
                stop_receiving = start_receiving + receiving_size 
                slices_rec.append(slice(start_receiving, stop_receiving))
                cmin_diff.append(start_receiving)
            cmin_diff = np.array(cmin_diff)
            slices_rec = tuple(slices_rec)
            slices = tuple(slices)
            
            if channel_axis is not None:
                size_with_channel = size[:channel_axis] + (n_channels,) + size[channel_axis:]
                slices_rec_with_channel = slices_rec[:channel_axis] + (slice(None),) + slices_rec[channel_axis:]
                slices_with_channel = slices[:channel_axis] + (slice(None),) + slices[channel_axis:]
            else:
                size_with_channel = size
                slices_rec_with_channel = slices_rec
                slices_with_channel = slices
            
            new_data = np.zeros(size_with_channel, dtype=np.uint16)

            new_data[slices_rec_with_channel] = receiving_layer.data
            
            receiving_layer.domain.coords_min = tuple(np.array(receiving_layer.domain.coords_min) - cmin_diff)
        else:
            new_data = receiving_layer.data
            
            slices = []
            for receiving_cmin, incoming_cmin, incoming_size in zip(
                receiving_layer.coords_min,
                incoming_layer.coords_min,
                incoming_layer.size,
            ):
                start = incoming_cmin - receiving_cmin
                stop = incoming_size + start
                slices.append(slice(start, stop))
            slices = tuple(slices)
            
            if channel_axis is not None:
                slices_with_channel = slices[:channel_axis] + (slice(None),) + slices[channel_axis:]
            else:
                slices_with_channel = slices

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

        new_data[slices_with_channel] = dst_arr

        receiving_layer.data = new_data
        receiving_layer.meta = incoming_layer.meta

    def on_first_merge(self, receiving_layer: Mask, incoming_layer: Mask):
        self.tile_tracker = InstanceTileTracker()

    def on_last_merge(self, receiving_layer: Mask, incoming_layer: Mask):
        receiving_layer.data = self.tile_tracker.resolve(receiving_layer.data)
        self.tile_tracker = InstanceTileTracker()
