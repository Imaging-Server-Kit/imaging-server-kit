from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np

from imaging_server_kit.types import DATA_TYPES, DataLayer
from imaging_server_kit.core.encoding import encode_contents


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
        self, layer_name: str, layer_data: Optional[np.ndarray], layer_meta: Dict
    ) -> Optional[DataLayer]: ...

    @abstractmethod
    def delete(self, layer_name: str): ...

    @abstractmethod
    def get_pixel_domain(self) -> np.ndarray: ...

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, idx: int) -> DataLayer:
        return self.layers[idx]

    def merge(
        self,
        layer_stack: Optional[LayerStackBase] = None,
        tiles_callback: Optional[Callable] = None,
    ):
        """Merge another layer stack, based on layer names.

        Parameters
        ----------
        layer_stack: Layer stack to be merged.
            Layers from this stack of the same kind, with the same name as instance layers (self.layers) will update corresponding meta and data attributes.
            Other layers from layer_stack will be added via the create() method.
        """
        if layer_stack is None:
            return

        for layer in layer_stack:
            existing_layer = self.read(layer.name)

            if existing_layer is None:
                initial_data = layer.get_initial_data()
                existing_layer = self.create(
                    layer.kind, initial_data, layer.name, layer.meta
                )
            else:
                if layer.is_first_tile:
                    self.update(
                        layer_name=existing_layer.name,
                        layer_data=layer.get_initial_data(),
                        layer_meta=layer.meta,
                    )

            if layer.is_tiled:
                existing_layer.merge_tile(layer.data, layer.meta)
                updated_data = existing_layer.data
                updated_meta = existing_layer.meta
            else:
                updated_data = layer.data
                updated_meta = layer.meta

            self.update(existing_layer.name, updated_data, updated_meta)

            if (layer.is_tiled) & (tiles_callback is not None):
                tiles_callback(
                    tile_idx=layer.meta["tile_params"]["tile_idx"],
                    n_tiles=layer.meta["tile_params"]["n_tiles"],
                ) # type: ignore

    def serialize(self, client_origin: str) -> List[Dict]:
        """Serialize a layer stack to JSON-compatible representation."""
        serialized_results = []
        for layer in self.layers:
            cls: Type[DataLayer] = DATA_TYPES[layer.kind]
            data = cls.serialize(layer.data, client_origin)
            meta = _serialize_meta(layer.meta)
            serialized_results.append(
                {
                    "kind": layer.kind,
                    "data": data,
                    "name": layer.name,
                    "meta": meta,
                }
            )
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

    def create(
        self,
        kind: str,
        data: Any,
        name: Optional[str] = None,
        meta: Optional[Dict] = None,
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
        layer = cls(name=name, data=data, meta=meta, **kwargs)

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

    def get_pixel_domain(self) -> np.ndarray:
        domains = []
        for data_layer in self.layers:
            domain = data_layer.pixel_domain()
            if domain is not None:
                domains.append(domain)
        if len(domains):
            # Final domain is the max bound of all parameter domains (Note: this assumes shared world coordinates, etc.)
            return np.max(np.stack(domains), axis=0)
        else:
            raise Exception("Could not compute pixel domain.")
