from typing import Dict, List, Optional
import numpy as np
from geojson import Feature, LineString

from imaging_server_kit.types.data_layer import DataLayer


def _preprocess_tile_info(current_data: np.ndarray, tile_info: dict):
    tile_info = tile_info.get("tile_params")
    ndim = tile_info.get("ndim")
    if ndim != current_data.shape[2]:
        raise Exception("Tile info does not match with data shape")

    tile_sizes = [tile_info.get(f"tile_size_{idx}") for idx in range(ndim)]
    tile_positions = [tile_info.get(f"pos_{idx}") for idx in range(ndim)]
    tile_max_positions = [
        tile_pos + tile_size
        for (tile_pos, tile_size) in zip(tile_positions, tile_sizes)
    ]

    coords = current_data[:, 0, :ndim][..., np.newaxis]
    mask = (coords >= tile_positions) & (
        coords < tile_max_positions
    )  # broadcasts to (N, ndim)
    all_filt = mask.all(axis=1)  # (N,)
    tile_data = current_data[all_filt].copy()  # Copy necessary?

    return ndim, tile_data, all_filt, tile_positions


class Vectors(DataLayer):
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
        self.kind = "vectors"
        self.dimensionality = (
            dimensionality if dimensionality is not None else list(np.arange(6))
        )
        if not required:
            self.default = None

    @classmethod
    def pixel_domain(cls, data: np.ndarray):
        return np.max(data[:, 0], axis=0)

    @classmethod
    def _get_tile(cls, current_data: np.ndarray, tile_info: dict) -> np.ndarray:
        ndim, tile_data, all_filt, tile_positions = _preprocess_tile_info(
            current_data, tile_info
        )
        tile_data[:, 0, :ndim] -= tile_positions
        return tile_data

    @classmethod
    def _merge_tile(
        cls, current_data: np.ndarray, tile_data: np.ndarray, tile_info: dict
    ) -> np.ndarray:
        ndim, tile_data, all_filt, tile_positions = _preprocess_tile_info(
            current_data, tile_info
        )
        tile_data[:, 0, :ndim] += tile_positions
        current_data = np.vstack((current_data[~all_filt], tile_data))
        return current_data

    @classmethod
    def to_features(cls, data):
        features = []
        vectors = data[:, :, ::-1]  # Invert XY
        for i, vector in enumerate(vectors):
            point_start = list(vector[0])
            point_end = list(vector[0] + vector[1])
            coords = [point_start, point_end]
            try:
                geom = LineString(coordinates=coords)
                features.append(Feature(geometry=geom, properties={"Detection ID": i}))
            except ValueError:
                print("Invalid line string geometry.")

        return features

    @classmethod
    def to_data(cls, features):
        vectors_arr = np.array(
            [feature["geometry"]["coordinates"] for feature in features]
        )
        origins = vectors_arr[:, 0]
        displacements = vectors_arr[:, 1] - origins
        vectors = np.stack((origins, displacements))
        vectors = np.rollaxis(vectors, 1)
        vectors = vectors[:, :, ::-1]  # Invert XY
        return vectors

    @classmethod
    def _get_initial_data(cls, pixel_domain):
        return np.zeros((0, 2, len(pixel_domain)), dtype=np.float32)

    @classmethod
    def validate_data(cls, data, meta):
        assert isinstance(
            data, np.ndarray
        ), f"Vectors data ({type(data)}) is not a Numpy array"
        assert len(data.shape) == 3, "Vectors data should have shape (N, 2, D)"
