from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import Domain
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
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            data=data,
            dimensionality=dimensionality,
            **kwargs,
        )

    @property
    def data_global_coords(self) -> Optional[np.ndarray]:
        """Data in global coordinate reference instead of local to the tile."""
        if (self.data is not None) and (self.domain is not None):
            if self.domain.coords_min is not None:
                data_global = self.data.copy()
                for dim in range(self.ndim):
                    data_global[:, dim] = (
                        data_global[:, dim] + self.domain.coords_min[dim]
                    )
                return data_global

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def _data_bounds(self) -> Optional[Tuple]:
        if self.data is not None:
            if self.n_objects > 0:
                return tuple(np.max(self.data, axis=0).tolist())

    def select(self, domain: Domain) -> Points:
        if self.data is None:
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self.initialize_data(domain=self.domain)
            _meta = self.meta
        else:
            # Select points via global coordinates
            points_in_tile = (self.data_global_coords >= domain.coords_min) & (
                self.data_global_coords < domain.coords_max
            )

            # All coordinates must be in the tile bounds
            tile_filter = points_in_tile.all(axis=1)  # (N,)

            # Select points in the tile
            points_tile_data = self.data[tile_filter]

            # Select meta of points in the tile
            points_tile_meta = extract_meta_tile(self.meta, self.n_objects, tile_filter)

            if len(points_tile_data) > 0:
                # Adjust the data so that the origin is zero at the origin of `domain`
                ptd = points_tile_data.copy()
                for dim in range(self.ndim):
                    ptd[:, dim] = ptd[:, dim] + (
                        self.domain.coords_min[dim] - domain.coords_min[dim]
                    )
                points_tile_data = ptd

            _data = points_tile_data
            _meta = points_tile_meta

        return Points(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta,
            domain=domain,
        )

    @staticmethod
    def initialize_data(domain: Optional[Domain]) -> Optional[np.ndarray]:
        if domain is None:
            return
        return np.zeros((0, domain.ndim), dtype=np.float32)
