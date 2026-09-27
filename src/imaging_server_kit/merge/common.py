import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from imaging_server_kit.core.domain import merge_domains
from imaging_server_kit.types.common import _get_slices_with_channel
from imaging_server_kit.types.layer import Layer


def _slices_from_origin(
    layer: Layer, origin: Sequence[float], channel_axis: Optional[int]
) -> Tuple[slice, ...]:
    """Slices covering the layer's extent, in an array whose first pixel is at `origin` (global coordinates)."""
    cmin_rounded = [math.floor(v - p) for v, p in zip(layer.coords_min, origin)]
    cmax_rounded = [math.ceil(v - p) for v, p in zip(layer.coords_max, origin)]
    return _get_slices_with_channel(cmin_rounded, cmax_rounded, channel_axis)


def prepare_canvas(
    receiving_layer: Layer, incoming_layer: Layer, dtype, copy_data: bool
) -> Tuple[np.ndarray, Tuple[slice, ...]]:
    """Array into which `incoming_layer` is merged, and the slices where it goes.

    If the incoming layer extends beyond the receiving layer, a zero-filled canvas of `dtype` covering both
    extents is created, the receiving data is painted into it, and `receiving_layer.position` is updated.
    Otherwise, the canvas is the receiving data itself (a `dtype` copy if `copy_data`).

    Note: `receiving_layer.data` is not updated; that's up to the merger.
    """
    channel_axis = incoming_layer.channel_axis

    merged_extent = merge_domains(
        domains=[receiving_layer.extent, incoming_layer.extent]
    )

    if merged_extent.size != receiving_layer.size:
        # The extent has changed: the receiving data is painted into a larger canvas
        origin = merged_extent.coords_min

        canvas_size = merged_extent.size
        if channel_axis is not None:
            n_channels = incoming_layer.shape[channel_axis]
            canvas_size = (
                canvas_size[:channel_axis] + (n_channels,) + canvas_size[channel_axis:]
            )

        canvas = np.zeros(tuple([math.ceil(v) for v in canvas_size]), dtype=dtype)
        canvas[_slices_from_origin(receiving_layer, origin, channel_axis)] = (
            receiving_layer.data
        )

        receiving_layer.position = origin
    else:
        # (Shortcut) The extent has not changed (incoming layer is fully contained in receiving layer)
        origin = receiving_layer.coords_min

        if copy_data:
            canvas = receiving_layer.data.astype(dtype)
        else:
            canvas = receiving_layer.data

    return canvas, _slices_from_origin(incoming_layer, origin, channel_axis)


def update_meta(
    receiving_layer: Layer,
    incoming_layer: Layer,
    exclude: Iterable[str] = ("position",),
) -> None:
    """Copy the incoming layer's meta into the receiving layer, except for the `exclude` keys."""
    for k, v in incoming_layer.meta.items():
        if k not in exclude:
            receiving_layer.meta[k] = v
