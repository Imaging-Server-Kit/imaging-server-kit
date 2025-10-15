from typing import Dict, List, Optional
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.data_layer import DataLayer


class Paths(DataLayer):
    def __init__(
        self,
        data: Optional[List] = None,
        name="Paths",
        description="Input paths shapes (2D, 3D)",
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
        self.kind = "paths"
        self.dimensionality = (
            dimensionality if dimensionality is not None else list(np.arange(6))
        )
        if not required:
            self.default = None

    @classmethod
    def pixel_domain(cls, data: np.ndarray):
        raise NotImplementedError("Not implemented")

    @classmethod
    def _get_tile(cls, current_data: np.ndarray, tile_info: dict) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    @classmethod
    def _merge_tile(
        cls, current_data: np.ndarray, tile_data: np.ndarray, tile_info: dict
    ) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    @classmethod
    def to_features(cls, data):
        return [encode_contents(arr.astype(np.float32) for arr in data)]

    @classmethod
    def to_data(cls, features):
        decoded = []
        for f in features:
            if isinstance(f, str):
                f = decode_contents(f)
            decoded.append(f.astype(float))
        return decoded

    @classmethod
    def _get_initial_data(cls, pixel_domain):
        raise NotImplementedError("Not implemented")
