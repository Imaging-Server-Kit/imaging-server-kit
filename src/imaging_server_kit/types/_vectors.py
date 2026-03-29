from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import Domain
from imaging_server_kit.types.data_layer import DataLayer
from imaging_server_kit.types.common import extract_meta_tile


class Vectors(DataLayer):
    """Data layer used to represent vectors.

    Parameters
    ----------
    data: A Numpy array of shape (N, 2, D) where D is the dimensionality (2, 3..).
        data[:, 0, :] represents the coordinates of the origin of the vectors.
        data[:, 1, :] represents the displacement from the origin.
    """

    kind = "vectors"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Vectors",
        description="Input vectors (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            data=data,
            description=description,
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
                    data_global[:, 0, dim] = (
                        data_global[:, 0, dim] + self.domain.coords_min[dim]
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
                return tuple(np.max(self.data[:, 0], axis=0))

    def select(self, domain: Domain) -> Vectors:
        if self.data is None:
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self.initialize_data(domain=self.domain)
            _meta = self.meta
        else:
            # Mask of vector coordinates in the tile
            vector_coords_in_tile = (
                self.data_global_coords[:, 0] >= domain.coords_min
            ) & (self.data_global_coords[:, 0] < domain.coords_max)

            # All coordinates must be in the tile bounds
            tile_filter = vector_coords_in_tile.all(axis=1)  # (N,)

            # Select vectors in the tile
            vectors_tile_data = self.data[tile_filter]

            # Select meta of vectors in the tile
            vectors_tile_meta = extract_meta_tile(
                self.meta, self.n_objects, tile_filter
            )

            if len(vectors_tile_data) > 0:
                vtd = vectors_tile_data.copy()
                for dim in range(self.ndim):
                    vtd[:, 0, dim] = vtd[:, 0, dim] + (
                        self.domain.coords_min[dim] - domain.coords_min[dim]
                    )
                vectors_tile_data = vtd

            _data = vectors_tile_data
            _meta = vectors_tile_meta

        return Vectors(
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
        return np.zeros((0, 2, domain.ndim), dtype=np.float32)
