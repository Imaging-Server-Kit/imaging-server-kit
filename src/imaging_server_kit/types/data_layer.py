from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import (
    TileMeta,
    TilingContext,
    generate_tiles,
    Domain,
)


class DataLayer:
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
        domain: Optional[Domain] = None,
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
        self._tile_meta = tile_meta

        # Assign a domain
        domain = Domain() if domain is None else domain.copy()
        if isinstance(translate, Tuple):
            domain.coords_min = translate
        self._domain = domain

        self._sync()

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

    def _sync(self):
        """Adjusts the domain to match new data set in the layer."""
        new_tile_meta = self.tile_meta.copy()
        new_domain = self.domain.copy()

        if self.bounds is None:
            new_domain.size = None
            new_domain.coords_min = None
        else:
            new_domain.size = self.bounds
            if new_domain.coords_min is None:
                new_domain.coords_min = tuple([0] * len(self.bounds))

            if new_tile_meta.overlap_px is None:
                new_tile_meta.overlap_px = tuple([0] * len(self.bounds))

        self._domain = new_domain
        self._tile_meta = new_tile_meta

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value
        self._sync()
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
    def tile_meta(self) -> TileMeta:
        return self._tile_meta

    @tile_meta.setter
    def tile_meta(self, value: TileMeta):
        self._tile_meta = value

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def ndim(self) -> Optional[int]:
        if self.domain is not None:
            return self.domain.ndim

    @property
    def size(self) -> Optional[Tuple]:
        if self.domain is not None:
            return self.domain.size

    @property
    def coords_min(self) -> Optional[Tuple]:
        if self.domain is not None:
            return self.domain.coords_min

    @property
    def coords_max(self) -> Optional[Tuple]:
        if self.domain is not None:
            return self.domain.coords_max

    @property
    def shape(self) -> Optional[Tuple]:
        if isinstance(self.data, np.ndarray):
            return self.data.shape

    @property
    def bounds(self) -> Optional[Tuple]:
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

    def select(self, domain: Domain) -> DataLayer:
        """Selection based on a domain in *global* coordinates."""
        cls = type(self)
        layer_selection = cls(
            data=self.data,
            name=self.name,
            meta=self.meta,
            tile_meta=self.tile_meta,
            domain=domain,
        )
        return layer_selection

    def __getitem__(self, key):
        """Selection based on a domain in *local* coordinates."""
        if not isinstance(key, tuple):
            key = (key,)

        position = []
        size = []
        for dim, k in enumerate(key):
            if isinstance(k, slice):
                start = 0 if k.start is None else k.start
                stop = self.bounds[dim] if k.stop is None else k.stop
                start_global = self.coords_min[dim] + start
                position.append(start_global)
                size.append(stop - start)
            else:
                # k is an `int`
                position.append(self.coords_min[dim] + k)
                size.append(0)

        if self.ndim is not None:
            if len(size) < self.ndim:
                for dim in range(len(size), self.ndim):
                    size.append(self.bounds[dim])
                    position.append(self.coords_min[dim])

        domain = Domain(position=position, size=size)

        return self.select(domain=domain)

    def reinitialize(self, domain: Domain) -> None:
        pass

    def zeros_in(self, domain: Optional[Domain]) -> Any:
        pass


class LayerTileGenerator:
    @staticmethod
    def generate_tiles(
        layer: DataLayer, ctx: Optional[TilingContext]
    ) -> Generator[DataLayer, None, None]:
        if ctx is None:
            yield layer.select(domain=Domain())
        else:
            for tile_meta, tile_domain in generate_tiles(
                domain=layer.domain,
                tile_size=ctx.tile_size,
                overlap=ctx.overlap,
                delay_sec=ctx.delay_sec,
                randomize=ctx.randomize,
            ):
                tile = layer.select(domain=tile_domain)
                tile.tile_meta = tile_meta
                yield tile
