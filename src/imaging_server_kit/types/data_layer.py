from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta, TilingContext, generate_tiles


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
    """

    kind: str = ""
    type = Union[str, np.ndarray, type(None)]

    def __init__(
        self,
        name: str = "",
        data: Any = None,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        translate: Optional[Tuple] = None,
        description: str = "",
        merger: str = "default",
        serializer: str = "default",
        validator: str = "default",
        **meta_kwargs,
    ):
        self._name = name

        # Prepare the meta attribute
        if meta is None:
            meta = {}
        else:
            meta["description"] = meta.get("description", description)

        # Convert dimensionality=None to the default 6-dims
        if "dimensionality" in meta_kwargs:
            if meta_kwargs.get("dimensionality") is None:
                meta_kwargs["dimensionality"] = np.arange(6).tolist()

        if "required" not in meta_kwargs:
            meta_kwargs["required"] = False

        # Add the meta kwargs
        # NOTE: this is important - only parameters passed to meta here get serialized
        for k, v in meta_kwargs.items():
            if not k in meta:
                meta[k] = v

        self._meta = meta

        # Handle required / default logic
        if meta_kwargs["required"] is True:
            if "default" in meta_kwargs:
                if data is None:
                    data = meta_kwargs["default"]
            else:
                if data is None:
                    raise ValueError(
                        f"`{name}` is required, but data is None and no defaults were given. \nEither set `required=False`, `default=...`, or `data=` to solve this issue.."
                    )

        self._data = data

        # Prepare the tile meta
        tile_meta = TileMeta() if tile_meta is None else tile_meta.copy()
        if isinstance(translate, Tuple):
            tile_meta.coords_min = translate
        self._tile_meta = tile_meta
        self._sync_tile_meta(self._tile_meta)

        # Merger
        self.merger = merger  # Merger `type`
        self.merger_instance = None  # Merger() object

        # Data serializer
        self.serializer = serializer
        self.serializer_instance = None

        # Validator
        self.validator = validator
        self.validator_instance = None

        # Run validation (`post-init`)
        if self.data is not None:
            # We can't do the import earlier.. is that a problem?
            from imaging_server_kit.validation.layer_validator import (
                find_layer_validator,
            )

            v = find_layer_validator(self)
            v.validate(self)

    def _sync_tile_meta(self, tile_meta: Optional[TileMeta]):
        new_tile_meta = TileMeta() if tile_meta is None else tile_meta.copy()

        if self.data_bounds is None:
            new_tile_meta.tile_size = None
            new_tile_meta.coords_min = None
        else:
            if tile_meta is None:
                _tile_pos = tuple([0] * len(self.data_bounds))
                _overlap_px = tuple([0] * len(self.data_bounds))
            else:
                _tile_pos = tile_meta.coords_min
                if _tile_pos is None:
                    _tile_pos = tuple([0] * len(self.data_bounds))
                _overlap_px = tile_meta.overlap_px
                if _overlap_px is None:
                    _overlap_px = tuple([0] * len(self.data_bounds))

            new_tile_meta.tile_size = self.data_bounds
            new_tile_meta.coords_min = _tile_pos
            new_tile_meta.overlap_px = _overlap_px

        self._tile_meta = new_tile_meta

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value
        # When the data changes, we synchronize the tile meta
        self._sync_tile_meta(self.tile_meta)
        self.refresh()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

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
    def bounds(self) -> Optional[Tuple]:
        if self.tile_meta is not None:
            return self.tile_meta.coords_max

    @property
    def data_bounds(self) -> Optional[Tuple]:
        pass

    @property
    def merger_instance(self):
        return self._merger

    @merger_instance.setter
    def merger_instance(self, value):
        self._merger = value

    def __str__(self) -> str:
        return f"{self.name} ({self.kind} layer). Data: {self.data.shape if isinstance(self.data, np.ndarray) else self.data}"

    def __repr__(self):
        return self.__str__()

    def refresh(self):
        pass

    def select(self, tile_meta: TileMeta) -> DataLayer:
        cls = type(self)
        layer_selection = cls(data=self.data, name=self.name, meta=self.meta, 
            tile_meta=tile_meta,  # TODO: Really? Problem is: we get first_tile=True when indexing..
        )
        return layer_selection

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        tile_size = []
        tile_pos = []
        for dim, k in enumerate(key):
            if isinstance(k, slice):
                start = 0 if k.start is None else k.start
                stop = self.bounds[dim] if k.stop is None else k.stop
                tile_pos.append(start)
                tile_size.append(stop - start)
            else:
                tile_pos.append(k)
                tile_size.append(0)

        if self.ndim is not None:
            if len(tile_size) < self.ndim:
                for dim in range(len(tile_size), self.ndim):
                    tile_size.append(self.bounds[dim])
                    tile_pos.append(self.tile_meta.coords_min[dim])

        tile_meta = TileMeta(
            tile_size=tile_size,
            tile_pos=tile_pos,
        )

        return self.select(tile_meta=tile_meta)

    @staticmethod
    def initialize_data(
        bounds: Optional[Union[List[int], Tuple]],
    ) -> Optional[np.ndarray]:
        pass


class LayerTileGenerator:
    @staticmethod
    def generate_tiles(
        layer: DataLayer, ctx: Optional[TilingContext]
    ) -> Generator[DataLayer, None, None]:
        if ctx is None:
            tile_meta = TileMeta()
            yield layer.select(tile_meta)
        else:
            for tile_meta in generate_tiles(
                bounds=layer.data_bounds,
                tile_size=ctx.tile_size,
                overlap=ctx.overlap,
                delay_sec=ctx.delay_sec,
                randomize=ctx.randomize,
            ):
                tile = layer.select(tile_meta)
                # Tiles inherit the global translation:
                tile.tile_meta.coords_min = layer.tile_meta.coords_min
                yield tile
