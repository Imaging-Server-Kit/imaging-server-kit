from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Type

import numpy as np

from imaging_server_kit.types import DATA_TYPES, DataLayer
from imaging_server_kit.core.tiling import TileMeta, TilingContext, generate_nd_tiles


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

    @property
    @abstractmethod
    def pixel_domain(self) -> Optional[List]: ...

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, idx: int) -> DataLayer:
        return self.layers[idx]

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

        dst_layers = []
        for src_layer in layer_stack:
            dst_layer = self.read(src_layer.name)
            if dst_layer is None:
                dst_layer = self.create(
                    kind=src_layer.kind,
                    name=src_layer.name,
                    meta=src_layer.meta,
                    merger=src_layer.merger_type,
                    data_serializer=src_layer.data_serializer_type,
                )
            dst_layers.append(dst_layer)

        for src_layer, dst_layer in zip(layer_stack, dst_layers):
            dst_layer.merge(src_layer)

    def serialize(self, client_origin: str) -> List[Dict]:
        """Serialize a layer stack to JSON-compatible representation."""
        serialized_results = []
        for layer in self.layers:
            serialized_layer = layer.serialize(client_origin)
            serialized_results.append(serialized_layer)
        return serialized_results

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

    def resolve_layer_name(self, kind: str, name: Optional[str] = None) -> str:
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

    def __init__(self, layers: Optional[List[DataLayer]] = None):
        self._layers: List[DataLayer] = []
        if layers is not None:
            for l in layers:
                self.create(
                    kind=l.kind,
                    data=l.data,
                    name=l.name,
                    meta=l.meta,
                    tile_meta=l.tile_meta,
                    merger=l.merger_type,
                    data_serializer=l.data_serializer_type,
                )

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
    def pixel_domain(self) -> Optional[List[int]]:
        domains = []
        for data_layer in self.layers:
            if data_layer.pixel_domain is not None:
                domains.append(data_layer.pixel_domain)
        if len(domains):
            # Final domain is the max bound of all parameter domains
            return np.max(np.stack(domains), axis=0).tolist()

    @classmethod
    def deserialize(cls, serialized_results: List[Dict], client_origin: str) -> Results:
        layers = []
        for serialized_layer in serialized_results:
            kind = serialized_layer["kind"]
            layer_cls: Type[DataLayer] = DATA_TYPES[kind]
            layer = layer_cls().deserialize(serialized_layer, client_origin)
            layers.append(layer)
        return Results(layers=layers)

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

        # Instantiate layer
        name = self.resolve_layer_name(kind, name)
        layer = cls(name=name, **kwargs)

        # Add layer to the stack
        self._layers.append(layer)

        return layer

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

    def generate_tiles(
        self, ctx: Optional[TilingContext]
    ) -> Generator[Results, None, None]:
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

    def get_tile(self, tile_meta: TileMeta) -> Results:
        return Results(layers=[l.get_tile(tile_meta.copy()) for l in self.layers])
