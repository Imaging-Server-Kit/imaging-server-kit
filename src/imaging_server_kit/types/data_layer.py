from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta, TilingContext, generate_nd_tiles


def resolve_tile_meta(
    _pixel_domain: Optional[Tuple], tile_meta: Optional[TileMeta]
) -> TileMeta:
    if _pixel_domain is None:
        _ndim = None
        _tile_size = None
        _tile_pos = None
        _overlap_px = None
    else:
        _ndim = len(_pixel_domain)
        _tile_size = _pixel_domain
        _tile_pos = tuple([0] * _ndim)
        _overlap_px = tuple([0] * _ndim)

    if tile_meta is None:
        return TileMeta(
            ndim=_ndim,
            tile_size=_tile_size,
            tile_pos=_tile_pos,
            overlap_px=_overlap_px,
            pixel_domain=_pixel_domain,
        )
    else:
        # If user has passed a tile_meta, check it against the layer data
        if tile_meta.ndim is None:
            tile_meta.ndim = _ndim
        if tile_meta.tile_size is None:
            tile_meta.tile_size = _tile_size
        if tile_meta.coords_min is None:
            tile_meta.coords_min = _tile_pos
        if tile_meta._pixel_domain is None:
            # Final pixel domain should be the max of _pixel_domain and _max_coords.
            if (tile_meta.coords_max is not None) and (_pixel_domain is not None):
                _pixel_domain_arr = np.asarray(_pixel_domain)
                _tile_meta_coords_max_arr = np.asarray(tile_meta.coords_max)
                _filt = _tile_meta_coords_max_arr > _pixel_domain_arr
                if _filt.sum() > 0:
                    _pixel_domain_arr[_filt] = _tile_meta_coords_max_arr[_filt]
                    _pixel_domain = _pixel_domain_arr.tolist()
            tile_meta.pixel_domain = _pixel_domain
        if tile_meta.overlap_px is None:
            tile_meta.overlap_px = _overlap_px

        return tile_meta


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

        self.tile_meta: TileMeta = resolve_tile_meta(self._pixel_domain, tile_meta)

        # Schema contributions
        self.constraints = [{}, {}]

    @property
    def shape(self) -> Optional[Tuple]:
        if isinstance(self.data, np.ndarray):
            return self.data.shape

    @property
    def ndim(self) -> Optional[int]:
        return self.tile_meta.ndim

    @property
    def tile_size(self) -> Optional[Tuple]:
        return self.tile_meta.tile_size

    @property
    def pixel_domain(self) -> Optional[Tuple]:
        return self.tile_meta.pixel_domain

    @property
    def _pixel_domain(self) -> Optional[Tuple]:
        """Pixel domain definition meant to be implemented by subclasses."""
        pass

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
    
    def first_tile_hook(self):
        pass
    
    def last_tile_hook(self):
        pass

    def merge(self, layer: DataLayer) -> None:
        self.data = layer.data
        self.meta = layer.meta

    def get_tile(self, tile_meta: TileMeta) -> DataLayer:
        cls = type(self)
        return cls(data=self.data, name=self.name, meta=self.meta, tile_meta=tile_meta)

    def generate_tiles(self, ctx: Optional[TilingContext]):
        if ctx is None:
            tile_meta = TileMeta()
            yield self.get_tile(tile_meta)
        else:
            for tile_meta in generate_nd_tiles(
                pixel_domain=self.pixel_domain,
                tile_size_px=ctx.tile_size_px,
                overlap_percent=ctx.overlap_percent,
                delay_sec=ctx.delay_sec,
                randomize=ctx.randomize,
            ):
                yield self.get_tile(tile_meta)

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
        cls, pixel_domain: Optional[Union[List[int], Tuple]]
    ) -> Optional[np.ndarray]:
        pass
