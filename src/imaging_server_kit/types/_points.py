from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.common import extract_meta_tile
from imaging_server_kit.types.data_layer import DataLayer


class Points(DataLayer):
    """Data layer used to represent points.

    Parameters
    ----------
    data: A Numpy array of shape (N, D) representing point coordinates, where D is the dimensionality (2, 3..).
    """

    kind = "points"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Points",
        description="Input points (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
            dimensionality=dimensionality,
            **kwargs,
        )

    @property
    def data_global_coords(self) -> Optional[np.ndarray]:
        """Data in global coordinate reference instead of local to the tile."""
        if (self.data is not None) and (self.tile_meta is not None):
            if self.tile_meta.coords_min is not None:
                data_global_coords = self.data.copy()
                for dim in range(self.ndim):
                    data_global_coords[:, dim] = (
                        data_global_coords[:, dim] + self.tile_meta.coords_min[dim]
                    )
                return data_global_coords

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def data_bounds(self) -> Optional[Tuple]:
        if self.data is not None:
            if self.n_objects > 0:
                return tuple(np.max(self.data, axis=0).tolist())

    def select(self, tile_meta: TileMeta) -> Points:
        if self.data is None:
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self.initialize_data(self.bounds)
            _meta = self.meta
        else:
            points_in_tile = (self.data >= tile_meta.coords_min) & (
                self.data < tile_meta.coords_max
            )

            # All coordinates must be in the tile bounds
            tile_filter = points_in_tile.all(axis=1)  # (N,)

            # Select points in the tile
            points_tile_data = self.data[tile_filter]

            # Select meta of points in the tile
            points_tile_meta = extract_meta_tile(self.meta, self.n_objects, tile_filter)

            if points_tile_data is not None:
                points_tile_data = points_tile_data - tile_meta.coords_min

            _data = points_tile_data
            _meta = points_tile_meta

        return Points(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=tile_meta,
        )

    @staticmethod
    def initialize_data(
        bounds: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if bounds is None:
            return
        return np.zeros((0, len(bounds)), dtype=np.float32)
