"""Common utilities for data layers (merging and accessing metadata in tiles, for Points, Vectors, Boxes, etc.)"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple
import numpy as np

if TYPE_CHECKING:
    from imaging_server_kit.core.domain import Domain
    from imaging_server_kit.types.layer import Layer


def _extract_meta(obj, n_objects, tile_filter):
    if isinstance(obj, Dict):
        return {k: _extract_meta(v, n_objects, tile_filter) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if len(obj) == n_objects:
            return obj[tile_filter]
    return obj


def select_object_meta(meta: Dict, n_objects: int, tile_filter: np.ndarray) -> Dict:
    """Iterates over two levels of the meta dictionary.
    Any numpy array found with length==n_objects is filtered using tile_filter.
    """
    if len(tile_filter) != n_objects:
        raise ValueError(
            f"tile_filter length ({len(tile_filter)}) does not match n_objects ({n_objects})"
        )
    
    return {k: _extract_meta(v, n_objects, tile_filter) for k, v in meta.items()}



def copy_meta(layer: Layer) -> Optional[Dict]:
    """Shallow copy of a layer's meta dictionary (None if the layer has no meta)."""
    return layer.meta.copy() if layer.meta is not None else None


def objects_in_domain(coords: np.ndarray, domain: Domain) -> np.ndarray:
    """Boolean filter (N,) of objects whose coordinates (N, ..., D) all lie in [coords_min, coords_max) of the domain.

    The upper bound is excluded (like tiles), so that objects on the border between two tiles are selected only once.
    """
    inside = (coords >= domain.coords_min) & (coords < domain.coords_max)
    return inside.reshape((len(coords), -1)).all(axis=1)

def _get_slices_with_channel(
    cmin_rounded: Sequence[int], cmax_rounded: Sequence[int], channel_axis: Optional[int]
) -> Tuple[slice, ...]:
    """Convenience function to get the slices, accounting for the channel axis."""
    slices = tuple(
        [slice(cmin, cmax) for cmin, cmax in zip(cmin_rounded, cmax_rounded)]
    )

    if channel_axis is not None:
        slices_with_channel = (
            slices[:channel_axis] + (slice(None),) + slices[channel_axis:]
        )
    else:
        slices_with_channel = slices

    return slices_with_channel


def domain_slices(
    layer: Layer, domain: Domain
) -> Optional[Tuple[Tuple[slice, ...], List[int]]]:
    """Slices selecting the intersection of `domain` (in global coordinates) with an array layer's data.

    Used by array layers (Image, Mask), which must provide a `channel_axis` property.

    Returns
    -------
    A tuple (slices_with_channel, cmin_rounded) where `cmin_rounded` is the position of the
    selection in global coordinates, or None if the domain does not intersect the layer.
    """
    extent = layer.extent

    cmin = [max(d, e) for d, e in zip(domain.coords_min, extent.coords_min)]
    cmax = [min(d, e) for d, e in zip(domain.coords_max, extent.coords_max)]

    csize = [c1 - c0 for c1, c0 in zip(cmax, cmin)]
    if any(size_i <= 0 for size_i in csize):
        # No intersection
        return None

    cmin_rounded = [math.floor(x) for x in cmin]

    starts, stops = [], []
    for cmin_i, size_i, extent_size_i, extent_cmin_i in zip(
        cmin_rounded, csize, extent.size, extent.coords_min
    ):
        # Make sure not to overflow the data (along the spatial axes)
        size_i = min(size_i, extent_size_i)
        s0 = int(cmin_i - extent_cmin_i)
        starts.append(s0)
        stops.append(s0 + int(size_i))

    slices_with_channel = _get_slices_with_channel(starts, stops, layer.channel_axis)

    return slices_with_channel, cmin_rounded
