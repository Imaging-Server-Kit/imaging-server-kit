from typing import Dict, List, Optional, Union
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.data_layer import DataLayer


class Tracks(DataLayer):
    """Data layer used to represent tracking data.

    Parameters
    ----------
    data: A Numpy array of shape (N, D+1) where the dimensions (D) are [ID, T, (Z), Y, X].
    """

    kind = "tracks"

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
        self.dimensionality = (
            dimensionality if dimensionality is not None else np.arange(6).tolist()
        )
        self.required = required

        # Schema contributions
        main = {}
        if not self.required:
            self.default = None
            main["default"] = self.default
        extra = {"dimensionality": self.dimensionality}
        self.constraints = [main, extra]

        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

        # TODO: Implement object-specific properties, like max_objects or min_track_length (could be validated).

    @classmethod
    def serialize(cls, data, client_origin: str) -> Optional[str]:
        if data is not None:
            return encode_contents(data.astype(np.float32))

    @classmethod
    def deserialize(cls, serialized_data: Optional[Union[np.ndarray, str]], client_origin: str) -> Optional[np.ndarray]:
        if serialized_data is None:
            return None
        if isinstance(serialized_data, str):
            serialized_data = decode_contents(serialized_data)
        return serialized_data.astype(float)

    @classmethod
    def _get_initial_data(cls, pixel_domain: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros((1, len(pixel_domain)+2), dtype=np.float32)

    def pixel_domain(self):
        if self.data is None:
            return
        return np.max(self.data, axis=0)[2:]
