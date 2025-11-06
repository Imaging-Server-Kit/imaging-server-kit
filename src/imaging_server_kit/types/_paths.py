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
        if self.data is None:
            return
        path_domains = []
        for path in self.data:
            path_domain = np.max(path, axis=0)
            path_domains.append(list(path_domain))
        path_domains = np.asarray(path_domains)
        pixel_domain = np.max(path_domains, axis=0)
        return pixel_domain

    @classmethod
    def serialize(cls, data: Optional[List[np.ndarray]], client_origin:str):
        if data is not None:
            return [encode_contents(arr.astype(np.float32)) for arr in data]

    @classmethod
    def deserialize(cls, serialized_data: Optional[str], client_origin: str) -> Optional[List[np.ndarray]]:
        if serialized_data is None:
            return None
        data = []
        for f in serialized_data:
            if isinstance(f, str):
                f = decode_contents(f)
                data.append(f.astype(float))
        return data

    @classmethod
    def _get_initial_data(cls, pixel_domain: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.asarray([])
