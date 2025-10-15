from typing import Dict, List, Optional
import numpy as np
from geojson import Feature, Polygon

from imaging_server_kit.types.data_layer import DataLayer


def _preprocess_tile_info(current_data: np.ndarray, tile_info: dict):
    tile_info = tile_info.get("tile_params")
    ndim = tile_info.get("ndim")

    tile_sizes = [tile_info.get(f"tile_size_{idx}") for idx in range(ndim)]
    tile_positions = [tile_info.get(f"pos_{idx}") for idx in range(ndim)]
    tile_max_positions = [
        tile_pos + tile_size
        for (tile_pos, tile_size) in zip(tile_positions, tile_sizes)
    ]

    if len(current_data):
        coords = np.asarray(current_data)[:, :, :ndim]
        mask = (coords >= tile_positions) & (coords < tile_max_positions)
        all_filt = mask.reshape((len(mask), -1)).all(axis=1)
        filtered_current_data = np.asarray(current_data)[~all_filt]
    else:
        filtered_current_data = current_data

    return ndim, filtered_current_data, tile_positions


class Boxes(DataLayer):
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Boxes",
        description="Input boxes shapes (2D, 3D)",
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
        self.kind = "boxes"
        self.dimensionality = (
            dimensionality if dimensionality is not None else list(np.arange(6))
        )
        if not required:
            self.default = None

    @classmethod
    def pixel_domain(cls, data: np.ndarray):
        return np.max(np.asarray(data), axis=(0, 1))

    @classmethod
    def _get_tile(cls, current_data: np.ndarray, tile_info: dict) -> np.ndarray:
        ndim, tile_data, tile_positions = _preprocess_tile_info(current_data, tile_info)
        tile_data[:, :, :ndim] -= tile_positions
        return tile_data

    @classmethod
    def _merge_tile(
        cls, current_data: np.ndarray, tile_data: np.ndarray, tile_info: dict
    ) -> np.ndarray:
        ndim, filtered_current_data, tile_positions = _preprocess_tile_info(
            current_data, tile_info
        )
        tile_data[:, :, :ndim] += tile_positions
        if len(filtered_current_data):
            current_data = np.vstack((filtered_current_data, tile_data))
        else:
            current_data = tile_data
        return current_data

    @classmethod
    def to_features(cls, data):
        features = []
        for i, box in enumerate(data):
            coords = np.array(box)[:, ::-1]  # Invert XY
            coords = coords.tolist()
            coords.append(coords[0])  # Close the Polygon
            try:
                geom = Polygon(coordinates=[coords], validate=True)
                features.append(Feature(geometry=geom, properties={"Detection ID": i}))
            except ValueError:
                print(
                    "Invalid box polygon geometry. Expected an array of shape (N, 4, D) representing the corners of the box."
                )
        return features

    @classmethod
    def to_data(cls, features):
        boxes = np.array([feature["geometry"]["coordinates"] for feature in features])
        boxes = np.array(
            [box[0] for box in boxes]
        )  # We get back a shape of (N, 1, 5, 2) - so we remove dim. 1
        boxes = boxes[:, :-1]  # Remove the last element
        boxes = boxes[:, :, ::-1]  # Invert XY
        return boxes

    @classmethod
    def _get_initial_data(cls, pixel_domain):
        return np.zeros((0, 4, len(pixel_domain)), dtype=np.float32)

    @classmethod
    def validate_data(cls, data, meta):
        assert isinstance(
            data, np.ndarray
        ), f"Boxes data ({type(data)}) is not a Numpy array"
        assert len(data.shape) == 3, "Boxes data should have shape (N, 4, D)"
