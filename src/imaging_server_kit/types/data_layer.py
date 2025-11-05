from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class DataLayer(ABC):
    """
    Data layer container for a particular data type.

    Attributes
    ----------
    data : None
        Data in the layer.
    name : str
        The name of the layer.
    description : str
        A short description of the layer.
    meta : dict
        Metadata about the layer.
    type : Any
        The type of data stored in the layer.
    kind : str
        A short string identifying the layer type.

    Methods
    -------
    update():
        Updates the data and meta attributes.
    validate_data():
        Validates a set of data and meta values.
    serialize():
        Serializes the class into a JSON-compatible representation.
    deserialize():
        Reconstructs an instance from a JSON representation.
    """

    kind: str = ""
    type = Union[str, np.ndarray, type(None)]

    def __init__(
        self,
        data=None,
        name: str = "",
        description: str = "",
        meta: Optional[Dict] = None,
    ):
        self.name = name
        self.description = description
        self.meta = meta if meta is not None else {}
        self.data = data

        # Schema contributions
        self.constraints = [{}, {}]

    @property
    def is_tiled(self) -> bool:
        return self.meta.get("tile_params") is not None

    @property
    def is_first_tile(self) -> bool:
        if not self.is_tiled:
            return False
        tile_params = self.meta["tile_params"]
        is_first_tile = tile_params.get("first_tile")
        return is_first_tile is not None

    def get_initial_data(self):
        # Figure out the pixel domain
        pixel_domain = self.pixel_domain()
        if self.is_tiled:
            tile_params = self.meta["tile_params"]
            ndim = tile_params["ndim"]
            if ndim is not None:
                pixel_domain = tuple(
                    [tile_params.get(f"domain_size_{idx}") for idx in range(ndim)]
                )
        return self._get_initial_data(pixel_domain) # type: ignore

    def __str__(self) -> str:
        return f"{self.name} ({self.kind} layer). Data: {self.data.shape if isinstance(self.data, np.ndarray) else self.data}"

    def __repr__(self):
        return self.__str__()

    def _validate(self, cls, v, meta, constraints):
        self.validate_data(v, meta, constraints)
        return v

    def update(self, updated_data: np.ndarray, updated_meta: Dict) -> None:
        self.data = updated_data
        self.meta = updated_meta
        self.refresh()

    def refresh(self):
        pass

    def pixel_domain(self) -> None:
        pass

    def merge_tile(self, tile_data: np.ndarray, tile_info: Dict):
        pass

    def get_tile(self, tile_info: Dict):
        return self.data, self.meta

    @classmethod
    def validate_data(cls, data: Any, meta: Dict, constraints: List[Dict]):
        pass

    @classmethod
    @abstractmethod
    def serialize(cls, data: Any, client_origin: str) -> Any: ...

    @classmethod
    @abstractmethod
    def deserialize(cls, serialized_data: Any, client_origin: str) -> Any: ...

    @classmethod
    def _get_initial_data(cls, pixel_domain: Optional[np.ndarray]) -> Optional[np.ndarray]:
        pass
