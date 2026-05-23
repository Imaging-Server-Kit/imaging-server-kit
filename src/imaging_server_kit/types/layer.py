from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.domain import Domain

from imaging_server_kit.core.tiling import (
    TileMeta,
    TilingSpecs,
    generate_tiles,
)


class Layer:
    """
    A layer corresponding to a particular kind of data.

    Attributes
    ----------
    data : Data in the layer.
    name : The name of the layer.
    meta : Metadata about the layer.
    merger: Merger strategy for the layer.
    description: Description of the layer.
    tile_meta : Tile metadata of the layer.
    position: Position of the layer.
    type : The type of data stored in the layer.
    kind : A short string identifying the layer type.
    extent : Extent (as a `domain`) of the layer.
    ndim : Dimensionality of the layer data.
    size : Size of the extent of the layer.
    coords_min : Minimum coordinates of the data (in world coordinates); given by the layer's extent.
    coords_max : Maximum coordinates of the data (in world coordinates); given by the layer's extent.
    shape : Data shape if it is an array-type.
    bounds : Size of the smallest spatial domain containing the data; should be implemented by subclasses.
    merger_instance : Merger instance associated with the layer.

    Methods
    ----------
    select() : Select data in the layer at the specified domain.
    refresh() : Refresh the layer's state.
    reiniitalize() : Reinitialize the specified domain in the layer; meant to be implemented by subclasses.
    zeros_in() : Provide zero-valued data in the specified domain; meant to be implemented by subclasses.
    """

    kind: str = ""
    type = Union[str, np.ndarray, type(None)]

    def __init__(
        self,
        name: str = "",
        data: Any = None,
        meta: Optional[Dict] = None,
        position: Optional[Tuple] = None,
        tile_meta: Optional[TileMeta] = None,
        description: str = "",
        merger: str = "default",
        **meta_kwargs,
    ):
        self._name = name

        # Prepare the meta attribute
        if meta is None:
            meta = {}

        meta["description"] = meta.get("description", description)
        meta["merger"] = meta.get("merger", merger)
        meta["position"] = meta.get("position", position)

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
        self._tile_meta = TileMeta() if tile_meta is None else tile_meta.copy()

        # Set the position attribute
        self._position = meta["position"]

        # Merger
        self.merger_instance = None

        # Run validation (`post-init`)
        if self.data is not None:
            # We can't do the import earlier.. is that a problem?
            from imaging_server_kit.validation.layer_validator import (
                find_layer_validator,
            )

            v = find_layer_validator(self)
            v.validate(self)

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value
        # self._sync()
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
    def position(self) -> Optional[Tuple]:
        if self._position is not None:
            return self._position
        else:
            if self.bounds is None:
                return
            else:
                return tuple([0] * len(self.bounds[0]))

    @position.setter
    def position(self, value):
        self._position = value
        self.refresh()  # Maybe needed (to check)

    @property
    def extent(self) -> Optional[Domain]:
        """Extent of the layer in global coordinates, as function of the data and position."""
        if self.bounds is None:
            return

        if self.position is None:
            return

        _coords_min, _coords_max = self.bounds
        _size = tuple([_max - _min for _max, _min in zip(_coords_max, _coords_min)])

        _position = tuple(
            [_cmin + _pos for _cmin, _pos in zip(_coords_min, self.position)]
        )

        return Domain(size=_size, position=_position)

    @property
    def ndim(self) -> Optional[int]:
        if self.extent is not None:
            return self.extent.ndim

    @property
    def size(self) -> Optional[Tuple]:
        if self.extent is not None:
            return self.extent.size

    @property
    def coords_min(self) -> Optional[Tuple]:
        if self.extent is not None:
            return self.extent.coords_min

    @property
    def coords_max(self) -> Optional[Tuple]:
        if self.extent is not None:
            return self.extent.coords_max

    @property
    def shape(self) -> Optional[Tuple]:
        if isinstance(self.data, np.ndarray):
            return self.data.shape

    @property
    def bounds(self) -> Optional[Tuple]:
        return None

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

    def select(self, domain: Domain) -> Layer:
        """Selection based on a domain in *global* coordinates."""
        cls = type(self)
        layer_selection = cls(
            data=self.data,
            name=self.name,
            meta=self.meta,
            tile_meta=self.tile_meta,
            position=domain.coords_min,  # Set the position to the domain's coords_min
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
                stop = self.size[dim] if k.stop is None else k.stop
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
                    size.append(self.size[dim])
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
        layer: Layer, ctx: Optional[TilingSpecs]
    ) -> Generator[Layer, None, None]:
        if ctx is None:
            yield layer.select(domain=Domain())
        else:
            for tile_meta, tile_domain in generate_tiles(
                domain=layer.extent,
                tile_size=ctx.tile_size,
                tile_overlap=ctx.tile_overlap,
                tile_delay=ctx.tile_delay,
                tile_randomize=ctx.tile_randomize,
            ):
                tile = layer.select(domain=tile_domain)
                tile.tile_meta = tile_meta
                yield tile
