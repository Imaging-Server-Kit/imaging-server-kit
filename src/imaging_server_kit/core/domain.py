from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

from imaging_server_kit.core._fmt import fmt_tuple


class Domain:
    """An n-D box defined by a size and a position in global pixel coordinates.

    Domains represent regions of interest and tile extents, and are used to restrict
    the computation of an algorithm to a region.

    Parameters
    ----------
    size : tuple, optional
        Size of the domain in pixels, one value per dimension.
    position : tuple, optional
        Position of the lower corner of the domain (top-left in 2D), in global pixel
        coordinates. Defaults to the origin if only `size` is provided.

    Attributes
    ----------
    size : tuple
        Size of the domain in pixels.
    coords_min : tuple
        Lower corner of the domain (top-left in 2D).
    coords_max : tuple
        Upper corner of the domain (bottom-right in 2D).
    ndim : int
        Number of dimensions.

    Examples
    --------
    >>> roi = sk.Domain(position=(20, 30), size=(60, 80))
    >>> roi.coords_max
    (80.0, 110.0)
    """

    def __init__(
        self,
        size: Optional[Union[Tuple, List]] = None,
        position: Optional[Union[Tuple, List]] = None,
    ):
        self._size = size

        # Position defaults to zero if only a size is specified
        if (size is not None) and (position is None):
            self._coords_min = tuple([0] * len(size))
        else:
            self._coords_min = position

    def __repr__(self) -> str:
        if self.size is None and self.coords_min is None:
            return "Domain(undefined)"
        return f"Domain(position={fmt_tuple(self.coords_min)}, size={fmt_tuple(self.size)})"

    @property
    def size(self) -> Optional[Tuple]:
        if self._size is not None:
            return tuple([float(v) for v in self._size])

    @size.setter
    def size(self, value: Optional[Tuple]):
        self._size = value

    @property
    def coords_min(self) -> Optional[Tuple]:
        if self._coords_min is not None:
            return tuple([float(v) for v in self._coords_min])

    @coords_min.setter
    def coords_min(self, value: Optional[Tuple]):
        self._coords_min = value

    @property
    def coords_max(self) -> Optional[Tuple]:
        if (self.coords_min is None) or (self.size is None):
            return

        return tuple(
            [
                float(coord_min_ax + size_ax)
                for (coord_min_ax, size_ax) in zip(self.coords_min, self.size)
            ]
        )

    @property
    def ndim(self) -> Optional[int]:
        if self.size is not None:
            return len(self.size)

    def _serialize(self) -> Dict:
        return {
            "position": self._coords_min,
            "size": self._size,
        }

    def _copy(self) -> Domain:
        return Domain(**self._serialize())

    def _merge(self, domain: Domain):
        if not all(
            [self.coords_max, self.coords_min, domain.coords_max, domain.coords_min]
        ):
            return

        new_coords_min = tuple(
            [min(a, b) for a, b in zip(self.coords_min, domain.coords_min)]
        )
        
        new_coords_max = tuple(
            [max(a, b) for a, b in zip(self.coords_max, domain.coords_max)]
        )

        new_size = tuple(
            [_max - _min for _max, _min in zip(new_coords_max, new_coords_min)]
        )

        self.coords_min = new_coords_min
        self.size = new_size


def merge_domains(domains: List[Optional[Domain]]) -> Optional[Domain]:
    """Create a new domain encompassing the extents of all provided domains.
    Domains with undefined size or position are ignored."""
    if len(domains) == 0:
        return

    elif len(domains) == 1:
        return domains[0]

    merged_domain = None
    for d in domains:
        if isinstance(d, Domain):
            merged_domain = d._copy()
            break

    if merged_domain is None:
        return merged_domain

    for d in domains:
        if isinstance(d, Domain):
            if all([d.coords_min, d.size]):
                merged_domain._merge(d)

    return merged_domain


def index_frame(origin: Optional[Tuple], extent: Optional[Domain]) -> Optional[Domain]:
    """Frame in which numpy-like indices are counted: from `origin` (e.g. a layer's position) to the end of `extent`.

    For object layers (Points, Boxes...) the extent is the bounding box of the data,
    so indices must be counted from the layer's position rather than from the extent's minimum.
    """
    if (origin is None) or (extent is None) or (extent.coords_max is None):
        return
    size = [max(cmax - o, 0) for cmax, o in zip(extent.coords_max, origin)]
    return Domain(size=size, position=origin)

def domain_from_key(key, extent: Optional[Domain]) -> Domain:
    """Domain (in global coordinates) selected by a numpy-like key (ints, slices, Ellipsis), relative to `extent`.

    Integer indices keep their dimension (size 1). Stepped slices are not supported.
    """
    if (extent is None) or (extent.size is None) or (extent.coords_min is None):
        raise IndexError("Cannot index spatially: undefined extent")

    if not isinstance(key, tuple):
        key = (key,)

    ndim = extent.ndim

    n_ellipsis = sum(k is Ellipsis for k in key)
    if n_ellipsis > 1:
        raise IndexError("An index can only have a single ellipsis ('...')")
    if n_ellipsis == 1:
        idx = key.index(Ellipsis)
        n_fill = ndim - (len(key) - 1)
        key = key[:idx] + (slice(None),) * max(n_fill, 0) + key[idx + 1 :]

    if len(key) > ndim:
        raise IndexError(
            f"Too many indices: extent is {ndim}-dimensional, but {len(key)} were indexed"
        )

    position = []
    size = []
    for dim, (cmin, n) in enumerate(zip(extent.coords_min, extent.size)):
        k = key[dim] if dim < len(key) else slice(None)
        if isinstance(k, slice):
            if k.step not in (None, 1):
                raise ValueError("Stepped slicing is not supported")
            start = _normalize_bound(k.start, n, default=0)
            stop = _normalize_bound(k.stop, n, default=n)
            stop = max(stop, start)
            position.append(cmin + start)
            size.append(stop - start)
        else:
            idx = k + n if k < 0 else k
            if not (0 <= idx < n):
                raise IndexError(
                    f"Index {k} is out of bounds for axis {dim} with size {n:g}"
                )
            position.append(cmin + idx)
            size.append(1)

    return Domain(size=size, position=position)


def _normalize_bound(value, n: float, default: float) -> float:
    """Normalize a slice bound like numpy: negative values count from the end, then clamp to [0, n]."""
    if value is None:
        return default
    if value < 0:
        value = value + n
    return min(max(value, 0), n)
