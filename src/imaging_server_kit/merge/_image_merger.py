from typing import Optional

import numpy as np

from imaging_server_kit.types._image import Image
from imaging_server_kit.merge.layer_merger import DefaultMerger


def overlap_count_map(layer: Image) -> Optional[np.ndarray]:
    """Return an array of the same shape as the tile containing the number of overlapping tiles at each pixel."""
    if (
        (layer.tile_meta.overlap_px is None)
        or (layer.size is None)
        or (layer.ndim is None)
    ):
        return

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
        layer.size, layer.tile_meta.overlap_px, first_tile_, last_tile_
    ):
        arr_i = np.arange(size_i)
        c = np.ones(size_i, dtype=np.int16)
        
        if not first_tile:
            c = c + (arr_i < overlap_i).astype(np.int16)
        
        if not last_tile:
            c = c + (arr_i >= size_i - overlap_i).astype(np.int16)
        
        out_shape = (1,) * len(per_axis) + (size_i,) + (1,) * (layer.ndim - len(per_axis) - 1)
        per_axis.append(c.reshape(out_shape))
    
    overlap_count_arr = np.ones(layer.size, dtype=np.int16)
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
            n_channels = incoming_layer.data.shape[channel_axis] 

        if (receiving_layer.data is None) or (receiving_layer.ndim is None):
            s = incoming_layer.size
            
            if channel_axis is not None:
                s_with_channel = s[:channel_axis] + (n_channels,) + s[channel_axis:]
            else:
                s_with_channel = s
            
            receiving_layer.data = np.zeros(s_with_channel, dtype=np.float32)
            
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
            
            new_data = np.zeros(size_with_channel, dtype=np.float32)

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

        _overlap_count_map = overlap_count_map(incoming_layer)
        
        # We `add` the incoming image data to merge it cleanly
        new_data[slices_with_channel] = new_data[slices_with_channel] + incoming_layer.data / _overlap_count_map
        
        receiving_layer.data = new_data
        receiving_layer.meta = incoming_layer.meta
