from typing import Dict, List, Optional
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.data_layer import DataLayer


class Paths(DataLayer):
    """Data layer used to represent 2D and 3D paths.

    Parameters
    ----------
    data: A list of Numpy arrays (one for each path), each with shape (N, D),
        where N is the length (number of points) in the path and D the dimensionality (2, 3..).
    """

    kind = "paths"

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
        self.dimensionality = (
            dimensionality if dimensionality is not None else np.arange(6).tolist()
        )
        self.required = required

        # Schema contributions
        main = {}
        if not self.required:
            main["default"] = None
        extra = {"dimensionality": self.dimensionality}
        self.constraints = [main, extra]

        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

        # TODO: Implement object-specific properties, like max_objects or max_path_length (could be validated).

    def pixel_domain(self):
        raise NotImplementedError("Not implemented")

    def get_tile(
        self, paths: np.ndarray, paths_meta: Dict, tile_info: Dict
    ) -> List[np.ndarray]:
        raise NotImplementedError("Not implemented")

    def merge_tile(self, paths_tile: np.ndarray, tile_info: Dict):
        raise NotImplementedError("Not implemented")

    @classmethod
    def serialize(cls, data, client_origin):
        return [encode_contents(arr.astype(np.float32) for arr in data)]

    @classmethod
    def deserialize(cls, serialized_data, client_origin):
        data = []
        for f in serialized_data:
            if isinstance(f, str):
                f = decode_contents(f)
            data.append(f.astype(float))
        return data

    @classmethod
    def _get_initial_data(cls, pixel_domain):
        raise NotImplementedError("Not implemented")
