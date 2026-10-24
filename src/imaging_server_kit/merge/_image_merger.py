import math
from typing import Optional

import numpy as np

from imaging_server_kit.types._image import Image
from imaging_server_kit.types.common import _get_slices_with_channel
from imaging_server_kit.merge.merger import DefaultMerger
from imaging_server_kit.merge.common import prepare_canvas, update_meta


class ImageTileOverlapMerger(DefaultMerger):
    """Merge images while averaging image intensities in overlapping regions.

    The merger counts how many tiles have covered each pixel of the receiving layer
    so far. Each incoming tile updates the running average of the pixels it covers,
    so that the merged image is correct after every tile, whatever the tile layout,
    overlap, or order. Pixels not covered by any tile of the current run (count of
    zero) are overwritten by the first tile that covers them.
    """

    def __init__(self):
        # Number of tiles merged at each pixel (same shape as the receiving data)
        self._counts: Optional[np.ndarray] = None

    def merge(self, receiving_layer: Image, incoming_layer: Image) -> None:
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.position is None):
            receiving_layer.position = incoming_layer.position
            receiving_layer.data = incoming_layer.data
            receiving_layer.meta = incoming_layer.meta
            self._counts = np.ones(incoming_layer.data.shape, dtype=np.int32)
            return

        if (self._counts is None) or (self._counts.shape != receiving_layer.data.shape):
            # The receiving data was not produced by this merge run
            self._counts = np.zeros(receiving_layer.data.shape, dtype=np.int32)

        old_coords_min = receiving_layer.coords_min
        old_coords_max = receiving_layer.coords_max

        new_data, slices_with_channel = prepare_canvas(
            receiving_layer, incoming_layer, dtype=np.float32, copy_data=True
        )

        if new_data.shape != self._counts.shape:
            # The canvas was enlarged: place the counts like the receiving data
            origin = receiving_layer.coords_min
            counts = np.zeros(new_data.shape, dtype=np.int32)
            counts_slices = _get_slices_with_channel(
                [math.floor(v - o) for v, o in zip(old_coords_min, origin)],
                [math.ceil(v - o) for v, o in zip(old_coords_max, origin)],
                incoming_layer.channel_axis,
            )
            counts[counts_slices] = self._counts
            self._counts = counts

        # Update the running average with the incoming tile
        counts = self._counts[slices_with_channel]
        new_data[slices_with_channel] = (
            new_data[slices_with_channel] * counts + incoming_layer.data
        ) / (counts + 1)
        self._counts[slices_with_channel] = counts + 1

        # Update the data of receiving layer
        receiving_layer.data = new_data

        # Meta becomes incoming layer's meta (except from position; we don't want to move the receiving layer)
        update_meta(receiving_layer, incoming_layer)

    def on_first_merge(self, receiving_layer: Image, incoming_layer: Image) -> None:
        if receiving_layer.data is None:
            self._counts = None
        elif receiving_layer is incoming_layer:
            # The layer was just added to the stack: its data counts as one tile
            self._counts = np.ones(receiving_layer.data.shape, dtype=np.int32)
        else:
            # Existing data is overwritten by the tiles of the new run
            self._counts = np.zeros(receiving_layer.data.shape, dtype=np.int32)
