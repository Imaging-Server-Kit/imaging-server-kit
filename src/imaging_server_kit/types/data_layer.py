from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta, TilingContext, generate_nd_tiles


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

        _pixel_domain = self._pixel_domain

        if _pixel_domain is not None:
            _ndim = len(_pixel_domain)
            _tile_size = _pixel_domain
            _tile_pos = [0] * _ndim
            _overlap_px = [0] * _ndim
        else:
            _ndim = None
            _tile_size = None
            _tile_pos = None
            _overlap_px = None

        if tile_meta is None:
            self.tile_meta = TileMeta(
                ndim=_ndim,
                tile_size=_tile_size,
                tile_pos=_tile_pos,
                overlap_px=_overlap_px,
                pixel_domain=_pixel_domain,
            )
        else:
            if self.data is not None:
                if tile_meta.ndim is None:
                    tile_meta.ndim = _ndim
                if tile_meta.shape is None:
                    tile_meta.shape = _tile_size
                if tile_meta.coords_min is None:
                    tile_meta.coords_min = _tile_pos
                if tile_meta._pixel_domain is None:
                    # Final pixel domain should be the max of _pixel_domain and _max_coords.
                    if (tile_meta.coords_max is not None) and (
                        _pixel_domain is not None
                    ):
                        _pixel_domain_arr = np.asarray(_pixel_domain)
                        _tile_meta_coords_max_arr = np.asarray(tile_meta.coords_max)
                        _filt = _tile_meta_coords_max_arr > _pixel_domain_arr
                        if _filt.sum() > 0:
                            _pixel_domain_arr[_filt] = _tile_meta_coords_max_arr[_filt]
                            _pixel_domain = _pixel_domain_arr.tolist()
                    tile_meta.pixel_domain = _pixel_domain
                if tile_meta.overlap_px is None:
                    tile_meta.overlap_px = _overlap_px

            self.tile_meta = tile_meta

        # Schema contributions
        self.constraints = [{}, {}]

    @property
    def ndim(self) -> Optional[int]:
        return self.tile_meta.ndim

    @property
    def shape(self) -> Optional[Tuple]:
        # TODO: Confusing; are we referring to the data shape, or to the shape of the bounds around the data?
        if isinstance(self.data, np.ndarray):
            return self.data.shape

    @property
    def pixel_domain(self) -> Optional[Tuple]:
        """The pixel domain of the layer is that of it's tile meta."""
        return self.tile_meta.pixel_domain

    @property
    def _pixel_domain(self) -> Optional[Tuple]:
        """Pixel domain definition meant to be implemented by subclasses."""
        pass

    # def get_initial_data(self):
    #     return self._get_initial_data(self.pixel_domain)

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
        # TODO: shouldn't this copy the tile_meta into a new instance of type(self) instead?
        cls = type(self)
        return cls(
            data=self.data,
            name=self.name,
            meta=self.meta,
            tile_meta=tile_meta  # Here...
        )
        # return self

    def generate_tiles(self, ctx: TilingContext):
        if self.pixel_domain is None:
            raise RuntimeError("Could not generate tiles; pixel domain is not defined.")

        for tile_meta in generate_nd_tiles(
            pixel_domain=self.pixel_domain,
            tile_size_px=ctx.tile_size_px,
            overlap_percent=ctx.overlap_percent,
            delay_sec=ctx.delay_sec,
            randomize=ctx.randomize,
        ):
            tile = self.get_tile(tile_meta)
            # tile_idx = tile_meta.tile_idx
            # n_tiles = tile_meta.n_tiles
            yield tile#, tile_idx, n_tiles

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
