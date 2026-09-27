from typing import Optional

import numpy as np

from imaging_server_kit.types._image import Image
from imaging_server_kit.merge.merger import DefaultMerger
from imaging_server_kit.merge.common import prepare_canvas, update_meta


def overlap_count_map(layer: Image) -> Optional[np.ndarray]:
    """Return an array of the same shape as the tile containing the number of overlapping tiles at each pixel."""
    if (layer.size is None) or (layer.ndim is None):
        return

    # If unspecified, overlap defaults to zero
    _overlap_px = layer.tile_meta.overlap_px
    if _overlap_px is None:
        if layer._bounds is not None:
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


def weighted_tile_data(layer: Image) -> np.ndarray:
    """Tile data divided by the number of overlapping tiles at each pixel.

    Summing the weighted tiles gives the average image intensity in overlapping regions.
    Tiles without overlap are returned as-is (their dtype is kept).
    """
    overlap_px = layer.tile_meta.overlap_px
    if (overlap_px is None) or not any(overlap_px):
        return layer.data

    return (layer.data / overlap_count_map(layer)).astype(np.float32)


class ImageTileOverlapMerger(DefaultMerger):
    """Merge images while averaging image intensities in overlapping regions."""

    @staticmethod
    def merge(receiving_layer: Image, incoming_layer: Image) -> None:
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.position is None):
            receiving_layer.position = incoming_layer.position
            receiving_layer.data = weighted_tile_data(incoming_layer)
            receiving_layer.meta = incoming_layer.meta
            return

        new_data, slices_with_channel = prepare_canvas(
            receiving_layer, incoming_layer, dtype=np.float32, copy_data=True
        )

        # We `add` the weighted incoming image data (average in overlapping regions)
        new_data[slices_with_channel] = new_data[
            slices_with_channel
        ] + weighted_tile_data(incoming_layer)

        # Update the data of receiving layer
        receiving_layer.data = new_data

        # Meta becomes incoming layer's meta (except from position; we don't want to move the receiving layer)
        update_meta(receiving_layer, incoming_layer)

    @staticmethod
    def on_first_merge(receiving_layer: Image, incoming_layer: Image) -> None:
        if (receiving_layer is incoming_layer) and (incoming_layer.data is not None):
            # The layer was just added to the stack (its data is not merged),
            # so its data must be weighted like that of the next tiles.
            receiving_layer.data = weighted_tile_data(incoming_layer)
