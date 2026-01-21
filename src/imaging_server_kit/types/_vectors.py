from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from geojson import Feature, LineString

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer, ObjectMerger
from imaging_server_kit.types.common import extract_meta_tile


def _get_tile(vectors: Vectors, tile_meta: TileMeta):
    # Mask of vector coordinates in the tile
    vector_coords_in_tile = (vectors.data[:, 0] >= tile_meta.coords_min) & (
        vectors.data[:, 0] < tile_meta.coords_max
    )

    # All coordinates must be in the tile bounds
    tile_filter = vector_coords_in_tile.all(axis=1)  # (N,)

    # Select vectors in the tile
    vectors_tile = vectors.data[tile_filter]

    # Select meta of vectors in the tile
    vectors_meta_tile = extract_meta_tile(vectors.meta, vectors.n_objects, tile_filter)

    return vectors_tile, vectors_meta_tile, tile_filter


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
        required: bool = True,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
        )
        self.dimensionality = (
            dimensionality if dimensionality is not None else np.arange(6).tolist()
        )
        self.required = required

        # Schema contributions
        main = {}
        if not self.required:
            self.default = None
            main["default"] = self.default
        extra = {
            "dimensionality": self.dimensionality,
            "required": self.required,
        }
        self.constraints = [main, extra]

        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)
            
        self.merger = ObjectMerger()

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
            vectors_tile_data, vectors_tile_meta, _ = _get_tile(self, tile_meta)
            if vectors_tile_data is not None:
                vectors_tile_data = vectors_tile_data - tile_meta.coords_min
            _data = vectors_tile_data
            _meta = vectors_tile_meta
        return Vectors(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=tile_meta,
        )

    @classmethod
    def serialize(
        cls, vectors: Optional[np.ndarray], client_origin: str
    ) -> Optional[List[Feature]]:
        if vectors is None:
            return None
        serialized_vectors = []
        vectors = vectors[:, :, ::-1]  # Invert XY
        for i, vector in enumerate(vectors):
            point_start = list(vector[0])
            point_end = list(vector[0] + vector[1])
            coords = [point_start, point_end]
            try:
                geom = LineString(coordinates=coords)
                serialized_vectors.append(
                    Feature(geometry=geom, properties={"Detection ID": i})
                )
            except ValueError:
                print("Invalid line string geometry.")
        return serialized_vectors

    @classmethod
    def deserialize(
        cls, serialized_vectors: Optional[List[Dict[str, Any]]], client_origin: str
    ) -> Optional[np.ndarray]:
        if serialized_vectors is None:
            return None

        vectors_arr = np.array(
            [feature["geometry"]["coordinates"] for feature in serialized_vectors]
        )
        vector_coords = vectors_arr[:, 0]
        displacements = vectors_arr[:, 1] - vector_coords
        vectors = np.stack((vector_coords, displacements))
        vectors = np.rollaxis(vectors, 1)
        vectors = vectors[:, :, ::-1]  # Invert XY
        return vectors

    @classmethod
    def _get_initial_data(
        cls, pixel_domain: Optional[Union[Tuple, List]]
    ) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        ndim = len(pixel_domain)
        return np.zeros((0, 2, ndim), dtype=np.float32)

    @classmethod
    def validate_data(cls, data, meta, constraints):
        main, extra = constraints
        if extra["required"] is False:
            return

        assert isinstance(
            data, np.ndarray
        ), f"Vectors data ({type(data)}) is not a Numpy array"
        assert len(data.shape) == 3, "Vectors data should have shape (N, 2, D)"
        assert data.shape[1] == 2, "Vectors data should have shape (N, 2, D)"
