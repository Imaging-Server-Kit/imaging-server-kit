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

    def pixel_domain(self):
        raise NotImplementedError("Not implemented")
    
    def get_tile(self, tile_info: Dict) -> List[np.ndarray]:
        raise NotImplementedError("Not implemented")

    def merge_tile(self, tracks_tile: np.ndarray, tile_info: Dict):
        raise NotImplementedError("Not implemented")

    @classmethod
    def serialize(cls, data, client_origin):
        return encode_contents(data.astype(np.float32))

    @classmethod
    def deserialize(cls, serialized_data: Union[np.ndarray, str], client_origin):
        if isinstance(serialized_data, str):
            serialized_data = decode_contents(serialized_data)
        return serialized_data.astype(float)

    @classmethod
    def _get_initial_data(cls, pixel_domain):
        raise NotImplementedError("Not implemented")
