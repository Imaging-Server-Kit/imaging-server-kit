from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.types.data_layer import DataLayer
from imaging_server_kit.core.tiling import Domain


class Image(DataLayer):
    """Data layer used to represent images and image-like data.

    Parameters
    ----------
    data: Numpy arrays.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    rgb: Set to True for RGB images.
    """

    kind = "image"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Image",
        description="Input image (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        rgb: bool = False,
        **kwargs,
    ):
        super().__init__(
            data=data,
            name=name,
            description=description,
            dimensionality=dimensionality,
            rgb=rgb,
            **kwargs,
        )

    @property
    def _data_bounds(self) -> Optional[Tuple]:
        if self._data is None:
            return
        if self.meta is None:
            return
        if self.meta["rgb"] is True:
            return (self._data.shape[0], self._data.shape[1])
        else:
            return self._data.shape

    def select(self, domain: Domain) -> Image:
        if (
            (self.data is None)
            or (domain.coords_max is None)
            or (self.coords_max is None)
        ):
            _data = None
        else:
            # TODO: Correct logic? We assume domain is global, and slices go from coords_min to coords_max
            domain_local = domain.copy()
            domain_local.coords_min = tuple(
                np.array(domain_local.coords_min) - np.array(self.coords_min)
            )
            # TODO: We should clip domain.slices or use zero-padding or sth before slicing:
            _data = self.data[domain_local.slices]

        return Image(
            data=_data,
            name=self.name,
            meta=self.meta,
            tile_meta=self.tile_meta,
            domain=domain,
        )

    @staticmethod
    def initialize_data(domain: Optional[Domain]) -> Optional[np.ndarray]:
        if domain is not None:
            return np.zeros(domain.size, dtype=np.float32)

    def initialize(self, domain_size: List[int]) -> Optional[np.ndarray]:
        if self.meta["rgb"] is True:
            domain_size.extend([3])
        return np.zeros(domain_size, dtype=np.float32)
