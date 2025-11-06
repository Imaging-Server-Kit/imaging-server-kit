from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from geojson import Feature, LineString

from imaging_server_kit.types.data_layer import DataLayer

from imaging_server_kit.types.common import merge_meta_tile, extract_meta_tile


def _preprocess_tile_info(vectors: np.ndarray, vectors_meta: Dict, tile_info: Dict):
    tile_info = tile_info["tile_params"]
    ndim = tile_info["ndim"]
    if ndim != vectors.shape[2]:
        raise ValueError(f"ndim from tile info ({ndim}) is incompatible with data shape ({vectors.shape})")

    tile_sizes = [tile_info[f"tile_size_{idx}"] for idx in range(ndim)]
    tile_positions = [tile_info[f"pos_{idx}"] for idx in range(ndim)]
    tile_max_positions = [
        tile_pos + tile_size
        for (tile_pos, tile_size) in zip(tile_positions, tile_sizes)
    ]

    n_objects = len(vectors)

    if n_objects:
        # Coordinates of the vectors
        vector_coords = vectors[:, 0, :ndim]

        # Mask of vector coordinates in the tile
        coords_in_tile = (vector_coords >= tile_positions) & (
            vector_coords < tile_max_positions
        )

        # All coordinates must be in the tile bounds
        tile_filter = coords_in_tile.all(axis=1)  # (N,)

        # Select vectors in the tile
        vectors_tile = vectors[tile_filter]

        # Select meta of vectors in the tile
        vectors_meta_tile = extract_meta_tile(vectors_meta, n_objects, tile_filter)
    else:
        vectors_tile = vectors
        vectors_meta_tile = vectors_meta
        tile_filter = np.full_like(vectors, fill_value=True, dtype=np.bool_)

    return ndim, vectors_tile, vectors_meta_tile, tile_filter, tile_positions


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
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
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

        # TODO: Implement object-specific properties, like max_objects or max_vector_length (could be validated).

    def pixel_domain(self):
        if self.data is None:
            return
        return np.max(self.data[:, 0], axis=0)

    def get_tile(self, tile_info: Dict) -> Tuple[np.ndarray, Dict]:
        ndim, vectors_tile, vectors_meta_tile, tile_filter, tile_positions = (
            _preprocess_tile_info(self.data, self.meta, tile_info)
        )
        # Offset the vectors in the tile by the tile position
        vectors_tile[:, 0, :ndim] = vectors_tile[:, 0, :ndim] - tile_positions
        return vectors_tile, vectors_meta_tile

    def merge_tile(self, vectors_tile: np.ndarray, tile_info: Dict):
        """Merges vectors from a tile into a set of existing vectors.
        Existing vectors inside the tile domain (from tile overlap) are replaced by vectors in the tile.
        """
        vectors_tile_meta = tile_info  # Long story..
        ndim, old_vectors_tile, old_vectors_meta_tile, tile_filter, tile_positions = (
            _preprocess_tile_info(self.data, vectors_tile_meta, tile_info)
        )

        # Offset the tile data by the tile positions
        vectors_tile[:, 0, :ndim] = vectors_tile[:, 0, :ndim] + tile_positions

        n_objects = len(self.data)

        if n_objects:
            # Remove the vectors from the vectors data that are in the tile
            vectors_clean = self.data[~tile_filter]

            # Merge the tile data with the cleaned vectors data
            merged_vectors = np.vstack((vectors_clean, vectors_tile))

            # Do the same for the meta
            merged_vectors_meta = merge_meta_tile(
                self.meta, vectors_tile_meta, n_objects, tile_filter
            )
        else:
            merged_vectors = vectors_tile
            merged_vectors_meta = vectors_tile_meta

        self.data = merged_vectors
        self.meta = merged_vectors_meta

    @classmethod
    def serialize(cls, vectors: Optional[np.ndarray], client_origin: str) -> Optional[List[Feature]]:
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
    def deserialize(cls, serialized_vectors: Optional[List[Dict[str, Any]]], client_origin: str) -> Optional[np.ndarray]:
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
    def _get_initial_data(cls, pixel_domain: Optional[np.ndarray]) -> Optional[np.ndarray]:
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
