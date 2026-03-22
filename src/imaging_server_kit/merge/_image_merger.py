import numpy as np

from imaging_server_kit.types._image import Image
from imaging_server_kit.merge.layer_merger import Merger


class ImageOverrideMerger(Merger):
    """Merge images using and `override` strategy: last tile overrides existing data in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Image, incoming_layer: Image) -> None:
        if (
            (incoming_layer.data is None)
            or (incoming_layer.tile_meta is None)
            or (incoming_layer.bounds is None)
        ):
            return

        if (
            (receiving_layer.data is None)
            or (receiving_layer.tile_meta is None)
            or (receiving_layer.bounds is None)
        ):
            receiving_layer.data = incoming_layer.initialize([1] * incoming_layer.ndim)
            receiving_layer.meta = incoming_layer.meta

        if (receiving_layer.bounds is None) or (receiving_layer.tile_meta is None):
            return  # This should never happen (just there for type hints)

        _slices = incoming_layer.tile_meta.slices
        if _slices is not None:
            _stack = np.stack([receiving_layer.bounds, incoming_layer.bounds])
            _bounds = np.max(_stack, axis=0).tolist()
            # If the incoming tile extends the pixel bounds, we create a new Image,
            # write receiving_layer.data into it, then merge the tile
            if _bounds != receiving_layer.bounds:
                new_data = incoming_layer.initialize(_bounds)
                new_data[receiving_layer.tile_meta.slices] = receiving_layer.data
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


class ImageTileOverlapMerger(Merger):
    """Merge images while averaging image intensities in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Image, incoming_layer: Image) -> None:
        if (
            (incoming_layer.data is None)
            or (incoming_layer.tile_meta is None)
            or (incoming_layer.bounds is None)
        ):
            return

        if (
            (receiving_layer.data is None)
            or (receiving_layer.tile_meta is None)
            or (receiving_layer.bounds is None)
        ):
            receiving_layer.data = incoming_layer.initialize([1] * incoming_layer.ndim)
            receiving_layer.meta = incoming_layer.meta

        if (receiving_layer.bounds is None) or (receiving_layer.tile_meta is None):
            return  # This should never happen (just there for type hints)

        _overlap_count_map = incoming_layer.tile_meta.overlap_count_map

        _slices = incoming_layer.tile_meta.slices
        if (_slices is not None) and (_overlap_count_map is not None):
            _stack = np.stack([receiving_layer.bounds, incoming_layer.bounds])
            _bounds = np.max(_stack, axis=0).tolist()  # (x, y)
            # If the incoming tile extends the pixel bounds, we create a new Image,
            # write receiving_layer.data into it, then merge the tile
            if _bounds != receiving_layer.bounds:
                new_data = incoming_layer.initialize(_bounds)
                new_data[receiving_layer.tile_meta.slices] = receiving_layer.data
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
