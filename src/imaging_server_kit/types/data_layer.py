from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta, generate_nd_tiles


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
        data: Any = None,
        name: str = "",
        description: str = "",
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
    ):
        self.name = name
        self.description = description
        self.meta = meta if meta is not None else {}
        self.data = data
        self.tile_meta = tile_meta

        # Schema contributions
        self.constraints = [{}, {}]

    @property
    def shape(self) -> Optional[Tuple]:
        if isinstance(self.data, np.ndarray):
            return self.data.shape
    
    @property
    def pixel_domain(self) -> Optional[Tuple]:
        pass

    @property
    def is_tile(self) -> bool:
        return self.tile_meta is not None

    def get_initial_data(self):
        if self.tile_meta is not None:
            pixel_domain = self.tile_meta.pixel_domain
        else:
            pixel_domain = self.pixel_domain
        return self._get_initial_data(pixel_domain)

    def __str__(self) -> str:
        return f"{self.name} ({self.kind} layer). Data: {self.data.shape if isinstance(self.data, np.ndarray) else self.data}"

    def __repr__(self):
        return self.__str__()

    def _validate(self, cls, v, meta, constraints):
        self.validate_data(v, meta, constraints)
        return v

    def update(self, updated_data: Any, updated_meta: Dict) -> None:
        self.data = updated_data
        self.meta = updated_meta
        self.refresh()

    def refresh(self):
        pass

    def merge_tile(self, tile: DataLayer) -> None:
        pass

    def get_tile(self, tile_meta: TileMeta) -> Optional[DataLayer]:
        return self

    def generate_tiles(self, tile_size_px, overlap_percent, delay_sec, randomize):
        if self.pixel_domain is None:
            raise RuntimeError("Could not generate tiles; pixel domain is not defined.")
        
        for tile_meta in generate_nd_tiles(
            pixel_domain=self.pixel_domain,
            tile_size_px=tile_size_px,
            overlap_percent=overlap_percent,
            delay_sec=delay_sec,
            randomize=randomize,
        ):
            tile = self.get_tile(tile_meta)
            tile_idx = tile_meta.tile_idx
            n_tiles = tile_meta.n_tiles
            yield tile, tile_idx, n_tiles

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
    def _get_initial_data(
        cls, pixel_domain: Optional[Union[List[int], Tuple[int]]]
    ) -> Optional[np.ndarray]:
        pass
