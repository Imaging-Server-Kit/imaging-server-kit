from typing import Optional

import numpy as np

from imaging_server_kit.types._image import Image
from imaging_server_kit.merge.layer_merger import DefaultMerger


class ImageOverrideMerger(DefaultMerger):
    """Merge images using and `override` strategy: last tile overrides existing data in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Image, incoming_layer: Image) -> None:
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.ndim is None):
            receiving_layer.data = incoming_layer.initialize([1] * incoming_layer.ndim)
            receiving_layer.domain.coords_min = incoming_layer.domain.coords_min

        min_bounds = np.min(
            np.stack([receiving_layer.coords_min, incoming_layer.coords_min]),
            axis=0,
        )

        max_bounds = np.max(
            np.stack([receiving_layer.coords_max, incoming_layer.coords_max]),
            axis=0,
        )

        size = max_bounds - min_bounds

        if tuple(size) != receiving_layer.size:
            new_data = incoming_layer.initialize(size)
            
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
            new_data[tuple(slices_rec)] = receiving_layer.data
            receiving_layer.domain.coords_min = tuple(np.array(receiving_layer.domain.coords_min) - np.array(cmin_diff))
        else:
            new_data = receiving_layer.data
            
            slices = []
            for receiving_cmin, incoming_cmin, incoming_size in zip(
                receiving_layer.coords_min,
                incoming_layer.coords_min,
                incoming_layer.size,
            ):
                diff = incoming_cmin - receiving_cmin
                start = diff
                stop = incoming_size + start
                slices.append(slice(start, stop))

        new_data[tuple(slices)] = incoming_layer.data

        receiving_layer.data = new_data
        receiving_layer.meta = incoming_layer.meta


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

    for n, ov, first_tile, last_tile in zip(
        layer.size, layer.tile_meta.overlap_px, first_tile_, last_tile_
    ):
        i = np.arange(n)
        c = np.ones(n, dtype=np.int16)
        if not first_tile:
            c = c + (i < ov).astype(np.int16)
        if not last_tile:
            c = c + (i >= n - ov).astype(np.int16)
        per_axis.append(
            c.reshape(
                (1,) * len(per_axis) + (n,) + (1,) * (layer.ndim - len(per_axis) - 1)
            )
        )

    overlap_count_arr = np.ones(layer.size, dtype=np.int16)
    for c in per_axis:
        overlap_count_arr *= c.reshape(
            c.shape
            + (1,)
            * (overlap_count_arr.ndim - c.ndim)  # Add fake dims (fixes the RGB case)
        )

    return overlap_count_arr


class ImageTileOverlapMerger(DefaultMerger):
    """Merge images while averaging image intensities in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Image, incoming_layer: Image) -> None:
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.ndim is None):
            receiving_layer.data = incoming_layer.initialize([1] * incoming_layer.ndim)
            receiving_layer.domain.coords_min = incoming_layer.domain.coords_min

        _overlap_count_map = overlap_count_map(incoming_layer)

        # Add fake dims (fixes the RGB case)
        _overlap_count_map_dims_matched = _overlap_count_map.reshape(
            _overlap_count_map.shape
            + (1,) * (incoming_layer.data.ndim - _overlap_count_map.ndim)
        )
        
        min_bounds = np.min(
            np.stack([receiving_layer.coords_min, incoming_layer.coords_min]),
            axis=0,
        )

        max_bounds = np.max(
            np.stack([receiving_layer.coords_max, incoming_layer.coords_max]),
            axis=0,
        )

        size = max_bounds - min_bounds

        if tuple(size) != receiving_layer.size:
            new_data = incoming_layer.initialize(size)
            
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
            new_data[tuple(slices_rec)] = receiving_layer.data
            receiving_layer.domain.coords_min = tuple(np.array(receiving_layer.domain.coords_min) - np.array(cmin_diff))
        else:
            new_data = receiving_layer.data
            
            slices = []
            for receiving_cmin, incoming_cmin, incoming_size in zip(
                receiving_layer.coords_min,
                incoming_layer.coords_min,
                incoming_layer.size,
            ):
                diff = incoming_cmin - receiving_cmin
                start = diff
                stop = incoming_size + start
                slices.append(slice(start, stop))

        # We `add` the incoming image data to merge it cleanly
        new_data[tuple(slices)] = new_data[tuple(slices)] + incoming_layer.data / _overlap_count_map_dims_matched

        receiving_layer.data = new_data
        receiving_layer.meta = incoming_layer.meta
