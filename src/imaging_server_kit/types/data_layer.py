from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta, TilingContext, generate_nd_tiles
from imaging_server_kit.types.common import merge_meta_tile


def multi_merge(layers: List[DataLayer]) -> DataLayer:
    if len(layers) == 0:
        raise ValueError("There should be at least one layers to merge.")
    elif len(layers) == 1:
        return layers[0]

    first_layer = layers[0]
    cls = type(first_layer)

    # Check that the items in `layers` are all of the same type
    for l in layers[1:]:
        if not isinstance(l, cls):
            raise ValueError("Layers to merge must be of the same type.")

    # Create a new instance of that type (TODO: or modify the first layer in-place?)
    # Could this be first_layer.copy()?
    merged_layer = cls(
        data=first_layer.data,
        name=first_layer.name,
        meta=first_layer.meta,
        tile_meta=first_layer.tile_meta,  # Correct?
    )
    
    for l in layers[1:]:
        merged_layer.merge(l)
    
    return merged_layer


class Merger(ABC):

    @abstractmethod
    def merge(self, src_layer: DataLayer, dst_layer: DataLayer) -> None:
        ...

    @abstractmethod
    def first_tile_hook(self, src_layer: DataLayer, dst_layer: DataLayer): ...

    @abstractmethod
    def last_tile_hook(self, src_layer: DataLayer, dst_layer: DataLayer): ...


class DefaultMerger(Merger):
    
    def merge(self, src_layer: DataLayer, dst_layer: DataLayer) -> None:
        src_layer.data = dst_layer.data
        src_layer.meta = dst_layer.meta
    
    def first_tile_hook(self, src_layer: DataLayer, dst_layer: DataLayer):
        pass

    def last_tile_hook(self, src_layer: DataLayer, dst_layer: DataLayer):
        pass


class ObjectMerger(DefaultMerger):

    def merge(self, src_layer: DataLayer, dst_layer: DataLayer) -> None:
        if (
            (dst_layer.data is None)
            or (dst_layer.tile_meta is None)
            or (dst_layer.pixel_domain is None)
        ):
            return

        if (
            (src_layer.data is None)
            or (src_layer.tile_meta is None)
            or (src_layer.pixel_domain is None)
        ):
            src_layer.data = dst_layer.data_global_coords
            src_layer.meta = dst_layer.meta
        else:
            if src_layer.n_objects > 0:
                merged_points_data = np.vstack((src_layer.data_global_coords, dst_layer.data_global_coords))
                merged_points_data = merged_points_data - src_layer.tile_meta.coords_min
                merged_points_meta = merge_meta_tile(
                    src_layer.meta, dst_layer.meta, src_layer.n_objects
                )
            else:
                merged_points_data = dst_layer.data
                merged_points_meta = dst_layer.meta

            src_layer.data = merged_points_data
            src_layer.meta = merged_points_meta
    
    def first_tile_hook(self, src_layer: DataLayer, dst_layer: DataLayer):
        # Erase all of the data before tiling
        src_layer.data = src_layer._get_initial_data(src_layer.data_pixel_domain)


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
        self._data = data
        self._name = name
        self._description = description
        self._meta = meta if meta is not None else {}
        
        self._tile_meta = tile_meta if tile_meta is None else tile_meta.copy()
        
        self._sync_tile_meta(self.tile_meta)

        # Schema contributions
        self.constraints = [{}, {}]
        
        self.merger = DefaultMerger()
    
    def _sync_tile_meta(self, tile_meta: Optional[TileMeta]):
        if self.data_pixel_domain is None:  # TODO: recursive issue with data_pixel_domain...
            new_tile_meta = tile_meta if tile_meta is not None else TileMeta()
            new_tile_meta.tile_size = None
            new_tile_meta.coords_min = None
            self._tile_meta = new_tile_meta
        else:
            _tile_size = self.data_pixel_domain
            if tile_meta is None:
                _tile_pos = tuple([0] * len(self.data_pixel_domain))
                _overlap_px = tuple([0] * len(self.data_pixel_domain))
            else:
                _tile_pos = tile_meta.coords_min
                if _tile_pos is None:
                    _tile_pos = tuple([0] * len(self.data_pixel_domain))
                _overlap_px = tile_meta.overlap_px
                if _overlap_px is None:
                    _overlap_px = tuple([0] * len(self.data_pixel_domain))
            
            new_tile_meta = tile_meta if tile_meta is not None else TileMeta()
            new_tile_meta.coords_min = _tile_pos
            new_tile_meta.tile_size = _tile_size
            new_tile_meta.overlap_px = _overlap_px
            self._tile_meta = new_tile_meta

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value
        self._sync_tile_meta(self.tile_meta)
        self.refresh()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def meta(self) -> Optional[Dict]:
        return self._meta

    @meta.setter
    def meta(self, value: Optional[Dict]):
        self._meta = value
        self.refresh()

    @property
    def tile_meta(self) -> Optional[TileMeta]:
        return self._tile_meta

    @property
    def shape(self) -> Optional[Tuple]:
        if isinstance(self.data, np.ndarray):
            return self.data.shape

    @property
    def ndim(self) -> Optional[int]:
        if self.tile_meta is not None:
            return self.tile_meta.ndim

    @property
    def tile_size(self) -> Optional[Tuple]:
        if self.tile_meta is not None:
            return self.tile_meta.tile_size

    @property
    def pixel_domain(self) -> Optional[Tuple]:
        if self.tile_meta is not None:
            return self.tile_meta.pixel_domain

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        """Pixel domain computed from the data, meant to be implemented by subclasses."""
        pass

    def __str__(self) -> str:
        return f"{self.name} ({self.kind} layer). Data: {self.data.shape if isinstance(self.data, np.ndarray) else self.data}"

    def __repr__(self):
        return self.__str__()

    def _validate(self, cls, v, meta, constraints):
        self.validate_data(v, meta, constraints)
        return v

    def refresh(self):
        pass

    def merge(self, layer: DataLayer) -> None:
        if layer.tile_meta.is_first_tile:
            self.merger.first_tile_hook(self, layer)
        self.merger.merge(self, layer)
        if layer.tile_meta.is_last_tile:
            self.merger.last_tile_hook(self, layer)

    def get_tile(self, tile_meta: TileMeta) -> DataLayer:
        cls = type(self)
        return cls(data=self.data, name=self.name, meta=self.meta, tile_meta=tile_meta)

    def generate_tiles(self, ctx: Optional[TilingContext]) -> Generator[DataLayer, None, None]:
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
