from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.types.data_layer import DataLayer
from imaging_server_kit.core.tiling import TileMeta


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
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        rgb: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
            description=description,
            dimensionality=dimensionality,
            rgb=rgb,
            **kwargs,
        )

    @property
    def data_bounds(self) -> Optional[Tuple]:
        if self._data is None:
            return
        if self.meta is None:
            return
        if self.meta["rgb"] is True:
            return (self._data.shape[0], self._data.shape[1])
        else:
            return self._data.shape

    def select(self, tile_meta: TileMeta) -> Image:
        if (
            (self.data is None)
            or (tile_meta.coords_max is None)
            or (self.bounds is None)
        ):
            _data = None
        else:
            _data = self.data[tile_meta.slices]

        return Image(
            data=_data,
            name=self.name,
            meta=self.meta,
            tile_meta=tile_meta,
        )

    @staticmethod
    def initialize_data(
        bounds: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if bounds is None:
            return
        return np.zeros(bounds, dtype=np.float32)

    def initialize(self, bounds: List[int]) -> Optional[np.ndarray]:
        if self.meta["rgb"] is True:
            bounds.extend([3])
        return self.initialize_data(bounds)
