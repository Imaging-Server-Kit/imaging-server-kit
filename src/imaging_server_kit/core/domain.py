from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union


class Domain:
    def __init__(
        self,
        position: Optional[Union[Tuple, List]] = None,
        size: Optional[Union[Tuple, List]] = None,
    ):
        self._size = size
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
            return tuple(self._size)

    @size.setter
    def size(self, value: Optional[Tuple]):
        self._size = value

    @property
    def coords_min(self) -> Optional[Tuple]:
        if self._coords_min is not None:
            return tuple(self._coords_min)

    @coords_min.setter
    def coords_min(self, value: Optional[Tuple]):
        self._coords_min = value

    @property
    def coords_max(self) -> Optional[Tuple]:
        if (self.coords_min is None) or (self.size is None):
            return

        return tuple(
            [
                coord_min_ax + size_ax
                for (coord_min_ax, size_ax) in zip(self.coords_min, self.size)
            ]
        )

    @property
    def ndim(self) -> Optional[int]:
        if self.coords_max is not None:
            return len(self.coords_max)

    @property
    def slices(self) -> Optional[Tuple]:
        if (self.coords_min is None) or (self.coords_max is None):
            return

        return tuple(
            [slice(cmin, cmax) for cmin, cmax in zip(self.coords_min, self.coords_max)]
        )

    def serialize(self) -> Dict:
        return {
            "position": self._coords_min,
            "size": self._size,
        }

    def copy(self) -> Domain:
        return Domain(**self.serialize())