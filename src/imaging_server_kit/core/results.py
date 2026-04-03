from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

import numpy as np

from imaging_server_kit.merge.layer_merger import LayerMerger
from imaging_server_kit.types import DATA_TYPES, DataLayer
from imaging_server_kit.core.tiling import (
    TileMeta,
    TilingContext,
    generate_tiles,
    Domain,
)


class LayerStackBase(ABC):
    """
    Base class representing a layer stack.
    """

    @property
    def layers(self):
        return []

    @abstractmethod
    def create(self, kind: str, name: Optional[str], **kwargs) -> DataLayer: ...

    @abstractmethod
    def read(self, name: str) -> Optional[DataLayer]: ...

    @abstractmethod
    def delete(self, name: str) -> None: ...

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, key) -> DataLayer:
        return self.layers[key]

    def merge(self, layer_stack: Optional[LayerStackBase]) -> None:
        """Merge another layer stack, based on layer names.

        Parameters
        ----------
        layer_stack: Layer stack to be merged.
            Layers from this stack of the same kind, with the same name as instance layers (self.layers)
            will update corresponding meta and data attributes.
            Other layers from layer_stack will be added via the create() method.
        """
        if layer_stack is None:
            return

        receiving_layers = []
        for incoming_layer in layer_stack:
            receiving_layer = self.read(incoming_layer.name)
            if receiving_layer is None:
                receiving_layer = self.create(
                    kind=incoming_layer.kind,
                    name=incoming_layer.name,
                    meta=incoming_layer.meta,
                    # TODO: correct?
                    tile_meta=incoming_layer.tile_meta,
                    domain=incoming_layer.domain,
                    merger=incoming_layer.merger,
                    serializer=incoming_layer.serializer,
                    
                    data=None,
                )
            receiving_layers.append(receiving_layer)

        layer_merger = LayerMerger()
        for incoming_layer, receiving_layer in zip(layer_stack, receiving_layers):
            layer_merger.merge(receiving_layer, incoming_layer)

        self.post_merge(receiving_layers)

    def post_merge(self, receiving_layers: List[DataLayer]):
        pass

    def to_params_dict(self) -> Dict[str, Any]:
        """
        Convert a layer stack to a dictionary representation mapping layer.name to layer.data.

        Examples
        ----------
        Use it to convert samples to runnable parameters:

        sample = algo.get_sample(0)
        params = sample.to_params_dict()
        results = algo.run(**params)
        """
        algo_params = {}
        for layer in self.layers:
            algo_params[layer.name] = layer.data
        return algo_params

    def _resolve_layer_name(self, kind: str, name: Optional[str] = None) -> str:
        # Make sure layer has a name
        if name is None:
            name = kind.capitalize()

        # Fix naming conflicts
        layer_names = [l.name for l in self.layers]
        name_idx = 1
        original_name = name
        while name in layer_names:
            name = f"{original_name}-{name_idx:02d}"
            name_idx += 1

        return name


class Results(LayerStackBase):
    """A stack of data layers.

    Access layers by index: `layer = results[0]` or name: `layer = results.read("Layer Name")`.

    Attributes
    ----------
    layers: List of layers in the stack.

    Methods
    ----------
    create(): Create a new layer.
    read(): Read a layer by name.
    delete(): Delete a layer by name.
    merge(): Merge another result stack, based on layer names.
        Layers of the same kind, with the same name will be updated (meta and data). Other layers will be added to the stack.
    """

    def __init__(
        self,
        layers: Optional[List[DataLayer]] = None,
        tile_meta: Optional[TileMeta] = None,
    ):
        self._layers: List[DataLayer] = []
        if layers is not None:
            for l in layers:
                self.create(
                    kind=l.kind,
                    data=l.data,
                    name=l.name,
                    meta=l.meta,
                    tile_meta=l.tile_meta,
                    domain=l.domain,
                    merger=l.merger,
                    serializer=l.serializer,
                )

        tile_meta = TileMeta() if tile_meta is None else tile_meta.copy()
        self._tile_meta = tile_meta

    def __str__(self):
        message = f"Results (Layers: {len(self.layers)})"
        for l in self.layers:
            message += "\n"
            message += l.__str__()
        return message

    def __repr__(self):
        return self.__str__()

    @property
    def layers(self) -> List[DataLayer]:
        return self._layers

    @property
    def tile_meta(self) -> TileMeta:
        return self._tile_meta

    @tile_meta.setter
    def tile_meta(self, value: TileMeta):
        self._tile_meta = value

    @property
    def ndim(self) -> Optional[int]:
        if self.coords_max is not None:
            return len(self.coords_max)

    @property
    def domain(self) -> Optional[Domain]:
        if (self.coords_max is not None) & (self.coords_min is not None):
            size = tuple(np.array(self.coords_max) - np.array(self.coords_min))
            return Domain(
                position=self.coords_min,
                size=size,
            )

    @property
    def coords_min(self) -> Optional[List[int]]:
        layer_coords_min = []
        for layer in self.layers:
            if layer.coords_min is not None:
                layer_coords_min.append(layer.coords_min)
        if len(layer_coords_min):
            return np.min(np.stack(layer_coords_min), axis=0).tolist()

    @property
    def coords_max(self) -> Optional[List[int]]:
        layer_coords_max = []
        for layer in self.layers:
            if layer.coords_max is not None:
                layer_coords_max.append(layer.coords_max)
        if len(layer_coords_max):
            # Final domain is the max of all layer bounds
            # TODO: Should we handle cases where, e.g. 2D and 3D data are both present?
            # For ex. by casting the lowest dimensionality layers to higher-dim?
            # For now, this is not supported (result layers must have ndim=None or all the same ndims).
            return np.max(np.stack(layer_coords_max), axis=0).tolist()

    def create(self, kind: str, name: Optional[str] = None, **kwargs) -> DataLayer:
        """Create a new layer in the results stack.

        Parameters
        ----------
        name: Name of the layer. If it already exists, a suffix will be added (e.g. Image-01).
        kind: The kind of layer: ["image", "mask", "points", "vectors", "tracks", "boxes", "paths", "float", "int", "bool", "str", "choice", "notification", "null"]
        """
        # Get layer class to instanciate
        cls: Optional[Type[DataLayer]] = DATA_TYPES.get(kind)
        if cls is None:
            raise ValueError(f"{kind} layers cannot be handled.")

        name = self._resolve_layer_name(kind, name)
        
        # Instantiate layer
        layer = cls(name=name, **kwargs)

        # Add layer to the stack
        self._layers.append(layer)

        self.post_create(layer)

        return layer

    def post_create(self, layer: DataLayer):
        pass

    def read(self, name: str) -> Optional[DataLayer]:
        """Read a layer by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer

    def delete(self, name: str) -> None:
        """Delete a layer by name."""
        for idx, layer in enumerate(self.layers):
            if layer.name == name:
                self._layers.pop(idx)

        self.post_delete(name)

    def post_delete(self, name: str) -> None:
        pass

    def select(self, domain: Domain) -> Results:
        return Results(layers=[l.select(domain=domain.copy()) for l in self.layers])

    def __getitem__(self, key) -> List[DataLayer]:
        # Results have a `layers` dimension (first dimension)
        # so we index as [Layer, Dim0, Dim1, .., DimN]
        if not isinstance(key, tuple):
            key = (key,)

        layer_key = key[0]

        if len(key) > 1:
            position = []
            size = []
            # We are selecting in the layers; we skip the `layers` dimension
            for dim, k in enumerate(key[1:]):
                if (
                    isinstance(k, slice)
                    & (self.coords_max is not None)
                    & (self.coords_min is not None)
                ):
                    start = (
                        self.coords_min[dim]
                        if k.start is None
                        else self.coords_min[dim] + k.start
                    )
                    stop = (
                        self.coords_max[dim]
                        if k.stop is None
                        else self.coords_min[dim] + k.stop
                    )
                    position.append(start)
                    size.append(stop - start)
                else:
                    position.append(self.coords_min[dim] + k)
                    size.append(1)

            if self.ndim is not None:
                if len(size) < self.ndim:
                    for dim in range(len(size), self.ndim):
                        position.append(self.coords_min[dim])
                        size.append(self.coords_max[dim] - self.coords_min[dim])

            domain = Domain(position=position, size=size)

            extract = self.select(domain=domain)
        else:
            extract = self

        if isinstance(layer_key, slice):
            start = 0 if layer_key.start is None else layer_key.start
            stop = len(self.layers) if layer_key.stop is None else layer_key.stop
            layer_selection = extract.layers[start:stop]
        else:
            # Layer_key is an int
            ## TODO: extract as a.. hard copy of the layers?
            layer_selection = extract.layers[layer_key]

        return layer_selection


class ResultsTileGenerator:
    @staticmethod
    def generate_tiles(
        results: Results, ctx: Optional[TilingContext]
    ) -> Generator[Results, None, None]:
        if ctx is None:
            tile_domain = Domain()
            yield results.select(domain=tile_domain)
        else:
            for tile_meta, tile_domain in generate_tiles(
                # TODO: This should be a sk.Domain?
                domain=results.domain,
                tile_size=ctx.tile_size,
                overlap=ctx.overlap,
                delay_sec=ctx.delay_sec,
                randomize=ctx.randomize,
            ):
                res_tile = results.select(domain=tile_domain)

                res_tile.tile_meta = tile_meta

                yield res_tile
