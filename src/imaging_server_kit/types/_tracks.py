from typing import Dict, List, Optional, Union
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.data_layer import DataLayer


class Tracks(DataLayer):
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Tracks",
        description="Input tracks (2D, 3D)",
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
        self.kind = "tracks"
        self.dimensionality = (
            dimensionality if dimensionality is not None else list(np.arange(6))
        )
        if not required:
            self.default = None

    @classmethod
    def pixel_domain(cls, data: np.ndarray):
        raise NotImplementedError("Not implemented")

    @classmethod
    def _get_tile(cls, self, current_data: np.ndarray, tile_info: dict) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    @classmethod
    def _merge_tile(
        cls, current_data: np.ndarray, tile_data: np.ndarray, tile_info: dict
    ) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    @classmethod
    def to_features(cls, data):
        return encode_contents(data.astype(np.float32))

    @classmethod
    def to_data(cls, features: Union[np.ndarray, str]):
        if isinstance(features, str):
            features = decode_contents(features)
        return features.astype(float)

    @classmethod
    def _get_initial_data(cls, pixel_domain):
        raise NotImplementedError("Not implemented")
