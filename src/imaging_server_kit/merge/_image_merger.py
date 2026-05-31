import math
from typing import Optional

import numpy as np

from imaging_server_kit.types._image import Image
from imaging_server_kit.merge.layer_merger import DefaultMerger
from imaging_server_kit.core.domain import merge_domains
from imaging_server_kit.merge.common import _get_slices_with_channel


def overlap_count_map(layer: Image) -> Optional[np.ndarray]:
    """Return an array of the same shape as the tile containing the number of overlapping tiles at each pixel."""
    if (layer.size is None) or (layer.ndim is None):
        return
    
    # If unspecified, overlap defaults to zero
    _overlap_px = layer.tile_meta.overlap_px
    if _overlap_px is None:
        if layer.bounds is not None:
            _overlap_px = tuple([0] * layer.ndim)

    per_axis = []

    if layer.tile_meta.first_tile is None:
        first_tile_ = [False] * layer.ndim
    else:
        first_tile_ = layer.tile_meta.first_tile

    if layer.tile_meta.last_tile is None:
        last_tile_ = [False] * layer.ndim
    else:
        last_tile_ = layer.tile_meta.last_tile

    for size_i, overlap_i, first_tile, last_tile in zip(
        layer.size, _overlap_px, first_tile_, last_tile_
    ):
        size_i = int(size_i)

        arr_i = np.arange(size_i)
        c = np.ones(size_i, dtype=np.int16)

        if not first_tile:
            c = c + (arr_i < overlap_i).astype(np.int16)

        if not last_tile:
            c = c + (arr_i >= size_i - overlap_i).astype(np.int16)

        out_shape = (
            (1,) * len(per_axis) + (size_i,) + (1,) * (layer.ndim - len(per_axis) - 1)
        )
        per_axis.append(c.reshape(out_shape))

    overlap_count_arr = np.ones([int(s) for s in layer.size], dtype=np.int16)
    for c in per_axis:
        overlap_count_arr *= c

    if layer.channel_axis is not None:
        # Expand the overlap array along the channel dimensions (repeat it n_channels times)
        channel_axis = layer.channel_axis
        n_channels = layer.data.shape[channel_axis]
        overlap_count_arr = np.expand_dims(overlap_count_arr, axis=channel_axis)
        overlap_count_arr = np.repeat(overlap_count_arr, n_channels, axis=channel_axis)

    return overlap_count_arr


class ImageTileOverlapMerger(DefaultMerger):
    """Merge images while averaging image intensities in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Image, incoming_layer: Image) -> None:
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
            new_data = np.zeros(size_with_channel, dtype=np.float32)
            
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

            new_data = receiving_layer.data.astype(np.float32)

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

        _overlap_count_map = overlap_count_map(incoming_layer)

        # We `add` the incoming image data to merge it cleanly with the overlap map
        new_data[slices_with_channel] = (
            new_data[slices_with_channel] + incoming_layer.data / _overlap_count_map
        )

        # Update the data of receiving layer
        receiving_layer.data = new_data

        # Meta becomes incoming layer's meta
        receiving_layer.meta = incoming_layer.meta

