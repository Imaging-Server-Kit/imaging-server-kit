import math
from typing import Optional

import numpy as np

from imaging_server_kit.merge.layer_merger import DefaultMerger
from imaging_server_kit.types._mask import Mask
from imaging_server_kit.core.domain import merge_domains
from imaging_server_kit.merge.common import _get_slices_with_channel
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
            n_channels = incoming_layer.shape[channel_axis] 

        if (receiving_layer.data is None) or (receiving_layer.position is None):
            receiving_layer.position = incoming_layer.position
            receiving_layer.data = incoming_layer.data
            receiving_layer.meta = incoming_layer.meta
            return

        merged_extent = merge_domains(
            domains=[receiving_layer.extent, incoming_layer.extent]
        )

        if merged_extent.size != receiving_layer.size:
            # Case where the extent has changed

            new_position = merged_extent.coords_min
            
            # Size with channel (not equivalent to .zeros_in() - TODO: but could it be implemented there?)
            if channel_axis is not None:
                size_with_channel = (
                    merged_extent.size[:channel_axis] + (n_channels,) + merged_extent.size[channel_axis:]
                )
            else:
                size_with_channel = merged_extent.size
            
            # Initialize new data array
            size_with_channel = tuple([math.ceil(v) for v in size_with_channel])
            new_data = np.zeros(size_with_channel, dtype=np.uint16)
            
            # Get the slice indices where to inpaint RECEIVING LAYER
            cmin_rounded = [
                math.floor(v - p)
                for v, p in zip(receiving_layer.coords_min, new_position)
            ]
            cmax_rounded = [
                math.ceil(v - p)
                for v, p in zip(receiving_layer.coords_max, new_position)
            ]
            
            slices_with_channel = _get_slices_with_channel(cmin_rounded, cmax_rounded, channel_axis)
            
            # Inpaint RECEIVING LAYER
            new_data[slices_with_channel] = receiving_layer.data
            
            # Update position
            receiving_layer.position = new_position
            
            # Get the slice indices where to inpaint INCOMING LAYER
            cmin_rounded = [
                math.floor(v - p)
                for v, p in zip(incoming_layer.coords_min, new_position)
            ]
            cmax_rounded = [
                math.ceil(v - p)
                for v, p in zip(incoming_layer.coords_max, new_position)
            ]
            
            slices_with_channel = _get_slices_with_channel(cmin_rounded, cmax_rounded, channel_axis)
        
        else:
            # (Shortcut) The extent has not changed (incoming layer is fully contained in receiving layer)
            
            new_data = receiving_layer.data

            # Get the slice indices where to inpaint incoming_layer
            cmin_rounded = [
                math.floor(v - p)
                for v, p in zip(incoming_layer.coords_min, receiving_layer.coords_min)
            ]
            cmax_rounded = [
                math.ceil(v - p)
                for v, p in zip(incoming_layer.coords_max, receiving_layer.coords_min)
            ]

            slices_with_channel = _get_slices_with_channel(cmin_rounded, cmax_rounded, channel_axis)
        
        # Simply override the data
        new_data[slices_with_channel] = incoming_layer.data

        # Update the data of receiving layer
        receiving_layer.data = new_data

        # Meta becomes incoming layer's meta
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
    
    # If unspecified, overlap defaults to zero
    _overlap_px = layer.tile_meta.overlap_px
    if _overlap_px is None:
        if layer.bounds is not None:
            _overlap_px = tuple([0] * len(layer.bounds))
    
    if (_overlap_px is None) or (layer.size is None):
        return

    size_int = tuple([math.ceil(v) for v in layer.size])

    overlap_slices = tuple(
        [
            slice(pos, max_pos - pos)
            for pos, max_pos in zip(_overlap_px, size_int)
        ]
    )
    
    mask = np.ones(size_int, dtype=np.uint8)
    
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

        if (receiving_layer.data is None) or (receiving_layer.position is None):
            receiving_layer.position = incoming_layer.position
            receiving_layer.data = incoming_layer.data
            receiving_layer.meta = incoming_layer.meta
            return
        
        merged_extent = merge_domains(
            domains=[receiving_layer.extent, incoming_layer.extent]
        )

        if merged_extent.size != receiving_layer.size:
            # Case where the extent has changed

            new_position = merged_extent.coords_min
            
            # Size with channel (not equivalent to .zeros_in() - TODO: but could it be implemented there?)
            if channel_axis is not None:
                size_with_channel = (
                    merged_extent.size[:channel_axis] + (n_channels,) + merged_extent.size[channel_axis:]
                )
            else:
                size_with_channel = merged_extent.size
            
            # Initialize new data array
            size_with_channel = tuple([math.ceil(v) for v in size_with_channel])
            new_data = np.zeros(size_with_channel, dtype=np.uint16)
            
            # Get the slice indices where to inpaint RECEIVING LAYER
            cmin_rounded = [
                math.floor(v - p)
                for v, p in zip(receiving_layer.coords_min, new_position)
            ]
            cmax_rounded = [
                math.ceil(v - p)
                for v, p in zip(receiving_layer.coords_max, new_position)
            ]
            
            slices_with_channel = _get_slices_with_channel(cmin_rounded, cmax_rounded, channel_axis)
            
            # Inpaint RECEIVING LAYER
            new_data[slices_with_channel] = receiving_layer.data
            
            # Update position
            receiving_layer.position = new_position
            
            # Get the slice indices where to inpaint INCOMING LAYER
            cmin_rounded = [
                math.floor(v - p)
                for v, p in zip(incoming_layer.coords_min, new_position)
            ]
            cmax_rounded = [
                math.ceil(v - p)
                for v, p in zip(incoming_layer.coords_max, new_position)
            ]
            
            slices_with_channel = _get_slices_with_channel(cmin_rounded, cmax_rounded, channel_axis)
        
        else:
            # (Shortcut) The extent has not changed (incoming layer is fully contained in receiving layer)
            
            new_data = receiving_layer.data

            # Get the slice indices where to inpaint incoming_layer
            cmin_rounded = [
                math.floor(v - p)
                for v, p in zip(incoming_layer.coords_min, receiving_layer.coords_min)
            ]
            cmax_rounded = [
                math.ceil(v - p)
                for v, p in zip(incoming_layer.coords_max, receiving_layer.coords_min)
            ]

            slices_with_channel = _get_slices_with_channel(cmin_rounded, cmax_rounded, channel_axis)


        receiving_layer.data = new_data  # Extend the source layer data


        src_tile = receiving_layer.select(domain=incoming_layer.extent)
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
        
        # Update the data of receiving layer
        receiving_layer.data = new_data

        # Meta becomes incoming layer's meta
        receiving_layer.meta = incoming_layer.meta

    def on_first_merge(self, receiving_layer: Mask, incoming_layer: Mask):
        self.tile_tracker = InstanceTileTracker()

    def on_last_merge(self, receiving_layer: Mask, incoming_layer: Mask):
        receiving_layer.data = self.tile_tracker.resolve(receiving_layer.data)
        self.tile_tracker = InstanceTileTracker()
