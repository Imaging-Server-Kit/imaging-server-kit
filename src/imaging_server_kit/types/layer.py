from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.domain import Domain, domain_from_key, index_frame
from imaging_server_kit.core._fmt import fmt_tuple, truncate
from imaging_server_kit.types.common import copy_meta

from imaging_server_kit.core.tiling import (
    TileMeta,
    TilingSpecs,
    generate_tiles,
)


def _build_meta(
    meta: Optional[Dict],
    description: str,
    merger: str,
    position: Optional[Tuple],
    meta_kwargs: Dict,
) -> Dict:
    """Merge the `meta` dictionary with the layer's keyword arguments (explicit `meta` keys take precedence)."""
    meta = dict(meta) if meta is not None else {}  # Copy: the caller's dict is not modified

    meta.setdefault("description", description)
    meta.setdefault("merger", merger)
    meta.setdefault("position", position)

    for k, v in meta_kwargs.items():
        if (k == "dimensionality") and (v is None):
            v = list(range(6))  # Convert dimensionality=None to the default 6-dims
        meta.setdefault(k, v)

    meta.setdefault("required", False)

    return meta


class Layer:
    """Base class of data layers.

    A data layer holds a single piece of data (an image, a mask, a numeric value,
    etc.) together with metadata. Layers represent both the inputs and the outputs of
    algorithms.

    Parameters
    ----------
    name : str, default=""
        Name of the layer.
    data : object, optional
        Data held by the layer. Its type depends on the subclass.
    meta : dict, optional
        Metadata dictionary. It is merged with `description`, `merger`, `position`,
        and extra keyword arguments (explicit `meta` keys take precedence).
    position : tuple, optional
        Position of the layer in global pixel coordinates.
    tile_meta : TileMeta, optional
        Metadata about the layer's position and role in a set of tiles.
    description : str, default=""
        Description of the layer, displayed on the algorithm documentation page.
    merger : str, default="default"
        Strategy used to merge the layer when it is assembled from tiles. Only `Mask`
        supports another strategy: `"instances"`.
    **meta_kwargs
        Extra metadata, added to `meta`. In Napari, metadata keys are applied as
        properties of the displayed layer (e.g. `colormap="viridis"`).

    Attributes
    ----------
    data
        Data held by the layer.
    name : str
        Name of the layer.
    meta : dict
        Metadata about the layer.
    kind : str
        A short string identifying the layer type, e.g. `"mask"`.
    position : tuple
        Position of the layer in global pixel coordinates.
    extent : Domain
        Region covered by the data in global pixel coordinates. For objects such as
        points, it is the bounding box of the objects.
    size : tuple
        Size of the extent.
    coords_min : tuple
        Lower corner of the extent.
    coords_max : tuple
        Upper corner of the extent.
    ndim : int
        Number of spatial dimensions.
    shape : tuple
        Shape of the data, if it is array-like.
    tile_meta : TileMeta
        Metadata about the layer's position and role in a set of tiles.
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

        # NOTE: this is important - only parameters stored in meta get serialized
        self._meta = _build_meta(meta, description, merger, position, meta_kwargs)

        # Handle required / default logic. The default is read from the merged meta,
        # so that the data and meta always agree.
        if (data is None) and (meta_kwargs.get("required", False) is True):
            if "default" not in self._meta:
                raise ValueError(
                    f"`{name}` is required, but data is None and no defaults were given. \nEither set `required=False`, `default=...`, or `data=` to solve this issue.."
                )
            data = self._meta["default"]

        self._data = data

        # Prepare the tile meta
        self._tile_meta = TileMeta() if tile_meta is None else tile_meta.copy()

        # Set the position attribute
        self._position = self._meta["position"]

        # Merger instance used while merging tiles into this layer (set by `LayerMerger`)
        self._merger_instance = None

        # Run validation (`post-init`)
        if self.data is not None:
            # Deferred import: the validation module imports the layer types (circular import)
            from imaging_server_kit.validation.layer_validator import (
                find_layer_validator,
            )

            find_layer_validator(self).validate(self)

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value
        self._refresh()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"Value must be str, got {type(value).__name__}")
        self._name = value

    @property
    def meta(self) -> Optional[Dict]:
        return self._meta

    @meta.setter
    def meta(self, value: Optional[Dict]):
        self._meta = value
        self._refresh()

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
            if self._bounds is None:
                return
            else:
                return tuple([0] * len(self._bounds[0]))

    @position.setter
    def position(self, value):
        self._position = value
        self.meta["position"] = value
        self._refresh()

    @property
    def extent(self) -> Optional[Domain]:
        """Extent of the layer in global coordinates, as function of the data and position."""
        if self._bounds is None:
            return

        if self.position is None:
            return

        _coords_min, _coords_max = self._bounds
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
    def _bounds(self) -> Optional[Tuple]:
        return None

    def __repr__(self) -> str:
        parts = [self._summary()]
        if self._position is not None and any(self._position):
            parts.append(f"at {fmt_tuple(self._position)}")
        if self.tile_meta.n_tiles > 1:
            parts.append(f"tile {self.tile_meta.tile_idx}/{self.tile_meta.n_tiles}")
        details = ", ".join(p for p in parts if p)
        return f"<{type(self).__name__} '{self.name}'{' ' + details if details else ''}>"

    def _summary(self) -> str:
        """Short description of the layer's data, used in `__repr__` and in the Stack table.

        Must stay cheap to compute: avoid full passes over large arrays."""
        data = self.data
        if data is None:
            return "empty"
        n_objects = getattr(self, "n_objects", None)
        if n_objects is not None:
            # Points, Boxes, Vectors, Paths
            summary = f"{n_objects} {self.kind}"
            if self.ndim is not None:
                summary += f", {self.ndim}D"
            return summary
        if isinstance(data, np.ndarray):
            return f"{data.dtype} {data.shape}"
        if isinstance(data, (bool, int, float, str, np.generic)):
            return f"= {truncate(repr(data))}"
        if isinstance(data, (list, tuple, dict)):
            return f"{type(data).__name__}[{len(data)}]"
        return f"= <{type(data).__name__}>"

    def _refresh(self):
        """Refresh the layer's state."""
        pass

    def _display(self) -> None:
        """Show the layer in the terminal, once each time it is merged into a Stack (e.g. notifications, progress bars).

        Meant to be implemented by subclasses; does nothing by default."""
        pass

    def select(self, domain: Domain) -> Layer:
        """Select the part of the layer inside a domain.

        Parameters
        ----------
        domain : Domain
            The region to select, in global pixel coordinates.

        Returns
        -------
        Layer
            A new layer with the selected data, positioned in global coordinates.
        """
        cls = type(self)
        
        required = True
        if self.meta:
            required=self.meta.get("required", True)
            
        _meta = copy_meta(self)
        
        layer_selection = cls(
            data=self.data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta.copy(),
            required=required
        )
        
        # Set the position to the domain's coords_min
        layer_selection.position = domain.coords_min
        
        return layer_selection

    def __getitem__(self, key):
        """Selection based on a numpy-like key in *local* coordinates (counted from the layer's position).

        Objects at negative local coordinates cannot be reached by indexing.
        """
        frame = index_frame(self.position, self.extent)
        return self.select(domain=domain_from_key(key, frame))

    def _reinitialize(self, domain: Domain) -> None:
        """Reinitialize the specified domain in the layer; meant to be implemented by subclasses."""
        pass

    def _zeros_in(self, domain: Optional[Domain]) -> Any:
        """Provide zero-valued data in the specified domain; meant to be implemented by subclasses."""
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
