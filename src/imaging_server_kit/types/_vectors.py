from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer, Merger
from imaging_server_kit.types.common import (
    extract_meta_tile,
    ObjectMerger,
    ObjectTileMerger,
)


class Vectors(DataLayer):
    """Data layer used to represent vectors.

    Parameters
    ----------
    data: A Numpy array of shape (N, 2, D) where D is the dimensionality (2, 3..).
        data[:, 0, :] represents the coordinates of the origin of the vectors.
        data[:, 1, :] represents the displacement from the origin.
    """

    kind = "vectors"
    mergers: Dict[str, Type[Merger]] = {
        "default": ObjectTileMerger,
        "override": ObjectMerger,
    }

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Vectors",
        description="Input vectors (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        merger: str = "default",
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            data=data,
            meta=meta,
            tile_meta=tile_meta,
            description=description,
            dimensionality=dimensionality,
            merger=merger,
            **kwargs,
        )

    @property
    def data_global_coords(self) -> Optional[np.ndarray]:
        """Data in global coordinate reference instead of local to the tile."""
        if (self.data is not None) and (self.tile_meta is not None):
            if self.tile_meta.coords_min is not None:
                data_global = self.data.copy()
                data_global[:, 0] = data_global[:, 0] + self.tile_meta.coords_min
                return data_global

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        if self.data is not None:
            if self.n_objects > 0:
                return tuple(np.max(self.data[:, 0], axis=0))

    def get_tile(self, tile_meta: TileMeta) -> Vectors:
        if self.data is None:
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self._get_initial_data(self.pixel_domain)
            _meta = self.meta
        else:
            # Mask of vector coordinates in the tile
            vector_coords_in_tile = (self.data[:, 0] >= tile_meta.coords_min) & (
                self.data[:, 0] < tile_meta.coords_max
            )

            # All coordinates must be in the tile bounds
            tile_filter = vector_coords_in_tile.all(axis=1)  # (N,)

            # Select vectors in the tile
            vectors_tile_data = self.data[tile_filter]

            # Select meta of vectors in the tile
            vectors_tile_meta = extract_meta_tile(self.meta, self.n_objects, tile_filter)
            
            if vectors_tile_data is not None:
                vectors_tile_data[:, 0] = vectors_tile_data[:, 0] - tile_meta.coords_min
            
            _data = vectors_tile_data
            _meta = vectors_tile_meta
        
        return Vectors(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=tile_meta,
        )

    @staticmethod
    def _get_initial_data(
        pixel_domain: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        ndim = len(pixel_domain)
        return np.zeros((0, 2, ndim), dtype=np.float32)