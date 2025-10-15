from typing import Dict, List, Optional, Union
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.data_layer import DataLayer


def _get_slices(current_data: np.ndarray, tile_info: dict):
    tile_info = tile_info.get("tile_params")
    ndim = tile_info.get("ndim")
    if ndim != current_data.ndim:
        raise Exception("Tile info does not match with data shape")

    tile_sizes = [tile_info.get(f"tile_size_{idx}") for idx in range(ndim)]
    tile_positions = [tile_info.get(f"pos_{idx}") for idx in range(ndim)]
    tile_max_positions = [
        tile_pos + tile_size
        for (tile_pos, tile_size) in zip(tile_positions, tile_sizes)
    ]
    slices = [
        slice(pos, max_pos) for pos, max_pos in zip(tile_positions, tile_max_positions)
    ]
    slices = tuple(slices)
    return slices


class Mask(DataLayer):
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Mask",
        description="Segmentation mask (2D, 3D)",
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
        self.kind = "mask"
        self.dimensionality = (
            dimensionality if dimensionality is not None else list(np.arange(6))
        )
        if not required:
            self.default = None

    @classmethod
    def pixel_domain(cls, data: np.ndarray):
        return data.shape

    @classmethod
    def _get_tile(cls, current_data: np.ndarray, tile_info: dict) -> np.ndarray:
        slices = _get_slices(current_data, tile_info)
        tile_data = current_data[slices]
        return tile_data

    @classmethod
    def _merge_tile(
        cls, current_data: np.ndarray, tile_data: np.ndarray, tile_info: dict
    ) -> np.ndarray:
        slices = _get_slices(current_data, tile_info)
        current_data[slices] = tile_data
        return current_data

    @classmethod
    def to_features(cls, data):
        return encode_contents(data.astype(np.uint16))

    @classmethod
    def to_data(cls, features: Union[np.ndarray, str]):
        if isinstance(features, str):
            features = decode_contents(features)
        return features.astype(int)

    @classmethod
    def _get_initial_data(cls, pixel_domain):
        return np.zeros(pixel_domain, dtype=np.uint16)

    @classmethod
    def validate_data(cls, data, meta):
        assert isinstance(
            data, np.ndarray
        ), f"Mask data ({type(data)}) is not a Numpy array"
