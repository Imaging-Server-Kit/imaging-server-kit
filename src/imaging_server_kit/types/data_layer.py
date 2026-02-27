from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta, TilingContext, generate_nd_tiles


def merge_layers(layers: List[DataLayer]) -> DataLayer:
    """Merge a list of data layers of the same kind.
    Note: This method differs from layer.merge(other_layer) which is an in-place merge.
    Here, a new layer is created and the data from all `layers` are merged into it.
    """
    if len(layers) == 0:
        raise ValueError("There should be at least one layer to merge.")
    elif len(layers) == 1:
        return layers[0]

    first_layer = layers[0]
    cls = type(first_layer)

    # Check that the items in `layers` are all of the same type
    for l in layers[1:]:
        if not isinstance(l, cls):
            raise ValueError("Layers to merge must be of the same type.")

    # Find the layers domain (note: same as Results.pixel_domain)
    domains = []
    for l in layers:
        if l.pixel_domain is not None:
            domains.append(l.pixel_domain)
    if len(domains):
        _domain = np.max(np.stack(domains), axis=0).tolist()

    # Create a new instance
    merged_layer = cls(
        data=cls._get_initial_data(_domain),
        name=first_layer.name,  # Use first layer name by convention
    )

    for l in layers:
        merged_layer.merge(l)

    return merged_layer


class Merger(ABC):
    @abstractmethod
    def merge(self, src_layer: DataLayer, dst_layer: DataLayer) -> None: ...

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
    """

    kind: str = ""
    type = Union[str, np.ndarray, type(None)]
    mergers: Dict[str, Type[Merger]] = {"default": DefaultMerger}

    def __init__(
        self,
        name: str = "",
        data: Any = None,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        translate: Optional[Tuple] = None,
        description: str = "",
        merger: Union[str, Merger] = "default",
        data_serializer: str = "default",
        **meta_kwargs,
    ):
        # self._data = data
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

        # Run data validation
        if data is not None:
            self.validate_data(data, self._meta)

        # Prepare the tile meta
        tile_meta = TileMeta() if tile_meta is None else tile_meta.copy()
        if isinstance(translate, Tuple):
            tile_meta.coords_min = translate
        self._tile_meta = tile_meta
        self._sync_tile_meta(self._tile_meta)

        # Merger
        if isinstance(merger, Merger):
            self.merger = merger
        else:
            # `merger` is a string; instanciate a Merger() from the available ones.
            if merger not in self.mergers:
                raise ValueError(
                    f"Merger `{merger}` is not supported. Available: {list(self.mergers.keys())}"
                )
            self.merger_type = merger
            merger_cls = self.mergers[merger]
            self.merger = merger_cls()

        # Data serializer
        self.data_serializer = data_serializer

    def _sync_tile_meta(self, tile_meta: Optional[TileMeta]):
        new_tile_meta = TileMeta() if tile_meta is None else tile_meta.copy()

        if self.data_pixel_domain is None:
            new_tile_meta.tile_size = None
            new_tile_meta.coords_min = None
        else:
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

            new_tile_meta.tile_size = self.data_pixel_domain
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
    def pixel_domain(self) -> Optional[Tuple]:
        if self.tile_meta is not None:
            return self.tile_meta.coords_max

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        """Pixel domain computed from the data, meant to be implemented by subclasses."""
        pass

    def __str__(self) -> str:
        return f"{self.name} ({self.kind} layer). Data: {self.data.shape if isinstance(self.data, np.ndarray) else self.data}"

    def __repr__(self):
        return self.__str__()

    def _validate(self, cls, v, meta):
        if v is not None:
            self.validate_data(v, meta)
        return v

    def refresh(self):
        pass

    def merge(self, layer: DataLayer) -> None:
        if not isinstance(layer.tile_meta, TileMeta):
            raise RuntimeError("Layer to merge has no tile meta.")
        if layer.tile_meta.is_first_tile:
            self.merger.first_tile_hook(self, layer)
        self.merger.merge(self, layer)
        if layer.tile_meta.is_last_tile:
            self.merger.last_tile_hook(self, layer)

    def get_tile(self, tile_meta: TileMeta) -> DataLayer:
        cls = type(self)
        return cls(data=self.data, name=self.name, meta=self.meta, tile_meta=tile_meta)

    def generate_tiles(
        self, ctx: Optional[TilingContext]
    ) -> Generator[DataLayer, None, None]:
        if ctx is None:
            tile_meta = TileMeta()
            yield self.get_tile(tile_meta)
        else:
            for tile_meta in generate_nd_tiles(
                pixel_domain=self.data_pixel_domain,
                tile_size_px=ctx.tile_size_px,
                overlap_percent=ctx.overlap_percent,
                delay_sec=ctx.delay_sec,
                randomize=ctx.randomize,
            ):

                tile = self.get_tile(tile_meta)

                # Tiles inherit the global translation:
                tile.tile_meta.coords_min = self.tile_meta.coords_min

                yield tile

    @staticmethod
    def validate_data(data: Any, meta: Dict):
        pass

    @staticmethod
    def _get_initial_data(
        pixel_domain: Optional[Union[List[int], Tuple]],
    ) -> Optional[np.ndarray]:
        pass
