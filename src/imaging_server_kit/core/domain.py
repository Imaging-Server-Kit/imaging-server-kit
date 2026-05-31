from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union


class Domain:
    """A nD-domain defined by a size and position in global pixel space.

    Attributes
    ----------
    size: Size of the domain in pixels.
    coords_min: The position of the top-left corner of the domain.
    coords_max: The position of the bottom-right corner of the domain.
    ndim: Number of dimensions.

    Methods
    ----------
    serialize(): Convert the domain to a dictionary format.
    copy(): Copy the domain.
    merge(): Merge another domain.
    """

    def __init__(
        self,
        size: Optional[Union[Tuple, List]] = None,
        position: Optional[Union[Tuple, List]] = None,
    ):
        self._size = size

        # Position defaults to zero if only a size is specified
        if (size is not None) & (position is None):
            self._coords_min = tuple([0] * len(size))
        else:
            self._coords_min = position

    def __str__(self):
        message = "Domain"
        message += "\n"
        message += f"Position: {self.coords_min}"
        message += "\n"
        message += f"Size: {self.size}"
        return message

    def __repr__(self):
        return self.__str__()

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
