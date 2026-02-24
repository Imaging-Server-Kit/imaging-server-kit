from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer
from imaging_server_kit.types.data_serializer import DataSerializer


class PathDataSerializer(DataSerializer):
    def serialize(self, paths: Optional[List[np.ndarray]], client_origin: str) -> Optional[List[str]]:
        if paths is not None:
            return [encode_contents(arr.astype(np.float32)) for arr in paths]
    
    def deserialize(self, serialized_paths: Optional[List[str]], client_origin: str) -> Optional[List[np.ndarray]]:
        if serialized_paths is None:
            return None
        data = []
        for f in serialized_paths:
            if isinstance(f, str):
                f = decode_contents(f)
                data.append(f.astype(float))
        return data


class Paths(DataLayer):
    """Data layer used to represent 2D and 3D paths.

    Parameters
    ----------
    data: A list of Numpy arrays (one for each path), each with shape (N, D),
        where N is the length (number of points) in the path and D the dimensionality (2, 3..).
    """

    kind = "paths"
    data_serializers: Dict[str, Type[DataSerializer]] = {"default": PathDataSerializer}

    def __init__(
        self,
        data: Optional[List] = None,
        name="Paths",
        description="Input paths shapes (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        data_serializer: str = "default",
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
            dimensionality=dimensionality,
            data_serializer=data_serializer,
            **kwargs,
        )
        
    def __str__(self) -> str:
        return f"{self.name} ({self.kind} layer). Paths: {self.n_objects}"

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        if self.data is None:
            return
        if self.n_objects > 0:
            path_domains = []
            for path in self.data:
                path_domain = np.max(path, axis=0)
                path_domains.append(list(path_domain))
            path_domains = np.asarray(path_domains)
            pixel_domain = np.max(path_domains, axis=0)
            return tuple(pixel_domain)

    @staticmethod
    def _get_initial_data(
        pixel_domain: Optional[Union[Tuple, List]]
    ) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.asarray([])
