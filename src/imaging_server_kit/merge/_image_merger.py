from typing import Optional

import numpy as np

from imaging_server_kit.types._image import Image
from imaging_server_kit.merge.layer_merger import Merger


class ImageOverrideMerger(Merger):
    """Merge images using and `override` strategy: last tile overrides existing data in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Image, incoming_layer: Image) -> None:
        if (incoming_layer.data is None) or (incoming_layer.coords_max is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.coords_max is None):
            receiving_layer.data = incoming_layer.initialize([1] * incoming_layer.ndim)
            receiving_layer.meta = incoming_layer.meta

        if receiving_layer.coords_max is None:
            return  # This should never happen (just there for type hints)

        _slices = incoming_layer.domain.slices
        if _slices is not None:
            _stack = np.stack([receiving_layer.coords_max, incoming_layer.coords_max])
            _bounds = np.max(_stack, axis=0).tolist()
            # If the incoming tile extends the pixel bounds, we create a new Image,
            # write receiving_layer.data into it, then merge the tile
            if _bounds != receiving_layer.coords_max:
                new_data = incoming_layer.initialize(_bounds)
                new_data[receiving_layer.domain.slices] = receiving_layer.data
            else:
                new_data = receiving_layer.data

            # New tile overrides existing data
            new_data[_slices] = incoming_layer.data  # type: ignore
            receiving_layer.data = new_data
            receiving_layer.meta = incoming_layer.meta

    @staticmethod
    def on_first_merge(receiving_layer: Image, incoming_layer: Image):
        receiving_layer.meta = incoming_layer.meta

    @staticmethod
    def on_last_merge(receiving_layer: Image, incoming_layer: Image):
        pass



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
        first_tile_ = [False]*layer.ndim
    else:
        first_tile_ = layer.tile_meta.first_tile
    
    if layer.tile_meta.last_tile is None:
        last_tile_ = [False]*layer.ndim
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
            * (
                overlap_count_arr.ndim - c.ndim
            )  # Add fake dims (fixes the RGB case)
        )

    return overlap_count_arr


class ImageTileOverlapMerger(Merger):
    """Merge images while averaging image intensities in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Image, incoming_layer: Image) -> None:
        if (incoming_layer.data is None) or (incoming_layer.coords_max is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.coords_max is None):
            receiving_layer.data = incoming_layer.initialize([1] * incoming_layer.ndim)
            receiving_layer.meta = incoming_layer.meta

        if receiving_layer.coords_max is None:
            return  # This should never happen (just there for type hints)

        _overlap_count_map = overlap_count_map(incoming_layer)

        _slices = incoming_layer.domain.slices
        if (_slices is not None) and (_overlap_count_map is not None):
            _stack = np.stack([receiving_layer.coords_max, incoming_layer.coords_max])
            _bounds = np.max(_stack, axis=0).tolist()  # (x, y)
            # If the incoming tile extends the pixel bounds, we create a new Image,
            # write receiving_layer.data into it, then merge the tile
            if _bounds != receiving_layer.coords_max:
                new_data = incoming_layer.initialize(_bounds)
                new_data[receiving_layer.domain.slices] = receiving_layer.data
            else:
                new_data = receiving_layer.data
            # We `add` the incoming image data to merge it cleanly
            # Add fake dims (fixes the RGB case)
            _overlap_count_map_dims_matched = _overlap_count_map.reshape(
                _overlap_count_map.shape
                + (1,) * (incoming_layer.data.ndim - _overlap_count_map.ndim)
            )
            new_data[_slices] = new_data[_slices] + incoming_layer.data / _overlap_count_map_dims_matched  # type: ignore
            receiving_layer.data = new_data
            receiving_layer.meta = incoming_layer.meta

    @staticmethod
    def on_first_merge(receiving_layer: Image, incoming_layer: Image):
        # Re-initialize image data on first tile to avoid accumulating data indefinitely on multiple runs
        receiving_layer.data = incoming_layer.initialize([1] * incoming_layer.ndim)
        receiving_layer.meta = incoming_layer.meta

    @staticmethod
    def on_last_merge(receiving_layer: Image, incoming_layer: Image):
        pass
