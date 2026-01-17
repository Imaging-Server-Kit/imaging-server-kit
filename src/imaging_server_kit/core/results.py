from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np

from imaging_server_kit.types import DATA_TYPES, DataLayer
from imaging_server_kit.core.encoding import encode_contents
from imaging_server_kit.core.tiling import TileMeta, TilingContext, generate_nd_tiles


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
        self,
        kind: str,
        data: Any,
        name: Optional[str],
        meta: Optional[Dict],
        tile_meta: Optional[TileMeta],
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

        for tile_layer in layer_stack:
            # Create a destination layer with data initialized like pixel_domain, if it does not exist
            dst_layer = self.read(tile_layer.name)
            if dst_layer is None:
                dst_layer = self.create(
                    kind=tile_layer.kind,
                    data=tile_layer._get_initial_data(tile_layer.pixel_domain),
                    name=tile_layer.name,
                    meta=tile_layer.meta,
                    tile_meta=None,  # Important!
                )

            # On first tile, erase the layer data to re-initialize it
            if tile_layer.tile_meta.is_first_tile:
                self.update(
                    layer_name=dst_layer.name,
                    updated_data=tile_layer._get_initial_data(tile_layer.pixel_domain),
                    updated_meta=tile_layer.meta,  # TODO: Not `initial_meta`?
                )
            
            # Merge the tile in the destination layer
            dst_layer.merge(tile_layer)

            # Trigger a layer update (will also do .refresh() and, for example, refresh the Napari viewer)
            self.update(
                layer_name=dst_layer.name,
                updated_data=dst_layer.data,
                updated_meta=dst_layer.meta,
            )

    def serialize(self, client_origin: str) -> List[Dict]:
        """Serialize a layer stack to JSON-compatible representation."""
        serialized_results = []
        for layer in self.layers:
            cls: Type[DataLayer] = DATA_TYPES[layer.kind]
            data = cls.serialize(layer.data, client_origin)
            meta = _serialize_meta(layer.meta)
            tile_meta = layer.tile_meta.serialize()
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
                self.create(l.kind, l.data, l.name, l.meta, l.tile_meta)

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
            # TODO: we should be able to track it via _update_result_pixel_domain()
            return np.max(np.stack(domains), axis=0).tolist()

    def _update_result_pixel_domain(self, layer: DataLayer):
        if (self.pixel_domain is not None) or (layer.pixel_domain is not None):
            if self.pixel_domain is None:
                for l in self.layers:
                    l.tile_meta.pixel_domain = layer.pixel_domain
            elif layer.pixel_domain is None:
                layer.tile_meta.pixel_domain = self.pixel_domain
            else:
                _pixel_domain_arr_results = np.asarray(self.pixel_domain)
                _pixel_domain_arr_layer = np.asarray(layer.pixel_domain)
                _filt = _pixel_domain_arr_results > _pixel_domain_arr_layer  # TODO: bug `operands could not be broadcast together with shapes (2,) (3,)`
                if _filt.sum() > 0:
                    _pixel_domain_arr_layer[_filt] = _pixel_domain_arr_results[_filt]
                    layer.tile_meta.pixel_domain = _pixel_domain_arr_layer.tolist()
                _filt = _pixel_domain_arr_results < _pixel_domain_arr_layer
                if _filt.sum() > 0:
                    _pixel_domain_arr_results[_filt] = _pixel_domain_arr_layer[_filt]
                    for l in self.layers:
                        l.tile_meta.pixel_domain = _pixel_domain_arr_results.tolist()

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

        # Match pixel domains
        self._update_result_pixel_domain(layer)

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
            self._update_result_pixel_domain(layer)
        return layer

    def delete(self, layer_name: str):
        """Delete a layer by name."""
        for idx, layer in enumerate(self.layers):
            if layer.name == layer_name:
                self._layers.pop(idx)
        # Recompute pixel domain for each layer
        for l in self.layers:
            l.tile_meta.pixel_domain = self.pixel_domain

    def generate_tiles(self, ctx: Optional[TilingContext]):
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
        return Results(layers=[l.get_tile(tile_meta) for l in self.layers])
