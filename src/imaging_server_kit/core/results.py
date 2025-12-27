from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np

from imaging_server_kit.types import DATA_TYPES, DataLayer
from imaging_server_kit.core.encoding import encode_contents
from imaging_server_kit.core.tiling import TileMeta, generate_nd_tiles


def _serialize_value(obj: Any) -> Any:
    if isinstance(obj, Dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return encode_contents(obj)
    return obj


def _serialize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively serialize Numpy arrays in the meta dictionary."""
    return {k: _serialize_value(v) for k, v in meta.items()}


class LayerStackBase(ABC):
    """
    Base class representing a layer stack.
    """

    @property
    def layers(self):
        return []

    @abstractmethod
    def create(
        self, kind: str, data: Any, name: Optional[str], meta: Optional[Dict]
    ) -> DataLayer: ...

    @abstractmethod
    def read(self, layer_name: str) -> Optional[DataLayer]: ...

    @abstractmethod
    def update(
        self, layer_name: str, updated_data: Optional[np.ndarray], updated_meta: Dict
    ) -> Optional[DataLayer]: ...

    @abstractmethod
    def delete(self, layer_name: str): ...

    @property
    @abstractmethod
    def pixel_domain(self) -> List[int]: ...

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, idx: int) -> DataLayer:
        return self.layers[idx]

    def _create_if_not_exists(self, layer: DataLayer) -> DataLayer:
        existing_layer = self.read(layer.name)
        if existing_layer is None:
            existing_layer = self.create(
                layer.kind,
                layer.get_initial_data(),
                layer.name,
                layer.meta,
            )
        return existing_layer

    def merge(self, layer_stack: LayerStackBase):
        """Merge another layer stack, based on layer names.

        Parameters
        ----------
        layer_stack: Layer stack to be merged.
            Layers from this stack of the same kind, with the same name as instance layers (self.layers)
            will update corresponding meta and data attributes.
            Other layers from layer_stack will be added via the create() method.
        """
        for layer in layer_stack:
            existing_layer = self._create_if_not_exists(layer)
            existing_layer.update(updated_data=layer.data, updated_meta=layer.meta)

    def serialize(self, client_origin: str) -> List[Dict]:
        """Serialize a layer stack to JSON-compatible representation."""
        serialized_results = []
        for layer in self.layers:
            cls: Type[DataLayer] = DATA_TYPES[layer.kind]
            data = cls.serialize(layer.data, client_origin)
            meta = _serialize_meta(layer.meta)
            tile_meta = _serialize_meta(layer.tile_meta)
            serialized_results.append(
                {
                    "kind": layer.kind,
                    "data": data,
                    "name": layer.name,
                    "meta": meta,
                    "tile_meta": tile_meta,
                }
            )
        return serialized_results

    def merge_tile(self, tile_results: LayerStackBase) -> None:
        for l in tile_results:
            if (l.tile_meta is None) or (l.is_tile is False):
                raise RuntimeError("Invalid attempt to merge a result tile.")

        for tile_layer in tile_results:
            layer = self._create_if_not_exists(tile_layer)
            if tile_layer.tile_meta.is_first_tile:
                self.update(
                    layer_name=layer.name,
                    updated_data=tile_layer.get_initial_data(),
                    updated_meta=tile_layer.meta,
                )
            layer.merge_tile(tile_layer)
            layer.refresh()
    
    

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
    update(): Update the data and meta attributes of a layer.
    delete(): Delete a layer by name.
    merge(): Merge another result stack, based on layer names.
        Layers of the same kind, with the same name will be updated (meta and data). Other layers will be added to the stack.
    """

    def __init__(self, layers: Optional[List[DataLayer]] = None):
        super().__init__()
        self._layers: List[DataLayer] = []
        if layers is not None:
            for l in layers:
                self.create(l.kind, l.data, l.name, l.meta)

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
    def pixel_domain(self) -> List[int]:
        domains = []
        for data_layer in self.layers:
            if data_layer.pixel_domain is not None:
                domains.append(data_layer.pixel_domain)
        if len(domains):
            # Final domain is the max bound of all parameter domains (Note: this assumes shared world coordinates, etc.)
            return np.max(np.stack(domains), axis=0).tolist()
        else:
            raise RuntimeError("Could not compute pixel domain.")

    def create(
        self,
        kind: str,
        data: Any,
        name: Optional[str] = None,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ) -> DataLayer:
        """Create a new layer in the results stack.

        Parameters
        ----------
        name: Name of the layer. If it already exists, a suffix will be added (e.g. Image-01).
        data: Data in the layer (must be compatible with the kind of layer).
        kind: The kind of layer: ["image", "mask", "points", "vectors", "tracks", "boxes", "paths", "float", "int", "bool", "str", "choice", "notification", "null"]
        meta: An optional dictionary of metadata about the layer.
        """
        # Make sure layer has a name
        if name is None:
            name = kind.capitalize()

        # Fix naming conflicts
        layer_names = [layer.name for layer in self.layers]
        name_idx = 1
        original_name = name
        while name in layer_names:
            name = f"{original_name}-{name_idx:02d}"
            name_idx += 1

        # Initialize meta if not provided
        if meta is None:
            meta = {}

        # Get layer class to instanciate
        cls: Optional[Type[DataLayer]] = DATA_TYPES.get(kind)
        if cls is None:
            raise ValueError(f"{kind} layers cannot be handled.")

        # Instantiate layer
        layer = cls(name=name, data=data, meta=meta, tile_meta=tile_meta, **kwargs)

        # Add layer to the stack
        self._layers.append(layer)

        return layer

    def read(self, layer_name: str) -> Optional[DataLayer]:
        """Read a layer by name."""
        for layer in self.layers:
            if layer.name == layer_name:
                return layer

    def update(
        self, layer_name: str, updated_data: Any, updated_meta: Dict
    ) -> Optional[DataLayer]:
        """Update the data and meta attributes of a layer."""
        layer = self.read(layer_name)
        if layer is not None:
            layer.update(updated_data, updated_meta)
        return layer

    def delete(self, layer_name: str):
        """Delete a layer by name."""
        for idx, layer in enumerate(self.layers):
            if layer.name == layer_name:
                self._layers.pop(idx)

    def generate_tiles(self, tile_size_px, overlap_percent, delay_sec, randomize):
        for tile_meta in generate_nd_tiles(
            pixel_domain=self.pixel_domain,
            tile_size_px=tile_size_px,
            overlap_percent=overlap_percent,
            delay_sec=delay_sec,
            randomize=randomize,
        ):
            tile_results = self.get_tile(tile_meta)
            tile_idx = tile_meta.tile_idx
            n_tiles = tile_meta.n_tiles
            yield tile_results, tile_idx, n_tiles

    def get_tile(self, tile_meta: TileMeta) -> Optional[Results]:
        tile_results = Results()
        for layer in self.layers:
            tile_layer = layer.get_tile(tile_meta)
            if tile_layer is None:
                return
            tile_results.create(
                name=tile_layer.name,
                kind=tile_layer.kind,
                data=tile_layer.data,
                meta=tile_layer.meta,
                tile_meta=tile_meta,
            )
        return tile_results
