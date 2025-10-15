from typing import Dict, List, Optional, Union
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.data_layer import DataLayer


def _preprocess_tile_info(current_data: np.ndarray, tile_info: dict):
    tile_info = tile_info.get("tile_params")
    ndim = tile_info.get("ndim")
    if ndim != current_data.shape[1]:
        raise Exception("Tile info does not match with data shape")

    tile_sizes = [tile_info.get(f"tile_size_{idx}") for idx in range(ndim)]
    tile_positions = [tile_info.get(f"pos_{idx}") for idx in range(ndim)]
    tile_max_positions = [
        tile_pos + tile_size
        for (tile_pos, tile_size) in zip(tile_positions, tile_sizes)
    ]

    coords = current_data[:, :ndim]  # shape (N, ndim)
    mask = (coords >= tile_positions) & (
        coords < tile_max_positions
    )  # broadcasts to (N, ndim)
    all_filt = mask.all(axis=1)  # (N,)
    tile_data = current_data[all_filt].copy()  # Copy necessary?

    return ndim, tile_data, all_filt, tile_positions


class Points(DataLayer):
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Points",
        description="Input points (2D, 3D)",
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
        self.kind = "points"
        self.dimensionality = (
            dimensionality if dimensionality is not None else list(np.arange(6))
        )
        if not required:
            self.default = None

    @classmethod
    def pixel_domain(cls, data: np.ndarray):
        return np.max(data, axis=0)

    @classmethod
    def _get_tile(cls, current_data: np.ndarray, tile_info: dict) -> np.ndarray:
        ndim, tile_data, all_filt, tile_positions = _preprocess_tile_info(
            current_data, tile_info
        )
        tile_data[:, :ndim] -= tile_positions
        return tile_data

    @classmethod
    def _merge_tile(
        cls, current_data: np.ndarray, tile_data: np.ndarray, tile_info: dict
    ) -> np.ndarray:
        ndim, tile_data, all_filt, tile_positions = _preprocess_tile_info(
            current_data, tile_info
        )
        tile_data[:, :ndim] += tile_positions
        current_data = np.vstack((current_data[~all_filt], tile_data))
        return current_data

    @classmethod
    def to_features(cls, data):
        # ndim = data.shape[1]
        # if ndim == 2:
        #     features = []
        #     point_coords = np.array(data)[:, ::-1]  # Invert XY
        #     for i, point in enumerate(point_coords):
        #         try:
        #             geom = Point(coordinates=[np.array(point).tolist()])
        #             features.append(
        #                 Feature(geometry=geom, properties={"Detection ID": i})
        #             )
        #         except:
        #             print("Invalid point geometry.")
        # elif ndim == 3:
        features = encode_contents(data.astype(np.float32))
        return features

    @classmethod
    def to_data(cls, features: Union[np.ndarray, str]):
        # if ndim == 2:
        #     # 2D case...
        #     if len(features):
        #         points = np.array(
        #             [feature["geometry"]["coordinates"] for feature in features]
        #         )
        #         points = points[:, 0, :]  # Remove an extra dimension
        #         points = points[:, ::-1]  # Invert XY
        #         return points
        #     else:
        #         return features
        # elif ndim == 3:
        #     # 3D case...
        if isinstance(features, str):
            features = decode_contents(features)
        return features.astype(float)

    @classmethod
    def _get_initial_data(cls, pixel_domain):
        return np.zeros((0, len(pixel_domain)), dtype=np.float32)

    @classmethod
    def validate_data(cls, data, meta):
        assert isinstance(
            data, np.ndarray
        ), f"Points data ({type(data)}) is not a Numpy array"
        assert len(data.shape) == 2, "Points data should have shape (N, D)"
