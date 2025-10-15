from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import numpy as np

from imaging_server_kit.types import DATA_TYPES


class LayerStackBase(ABC):
    """Implements CRUD + merge for data layers."""
    @abstractmethod
    def __iter__(self): ...

    @abstractmethod
    def __getitem__(self): ...

    @abstractmethod
    def create(self, kind, data, name: Optional[str], meta: Optional[Dict]): ...

    @abstractmethod
    def read(self, layer_name): ...

    @abstractmethod
    def update(self, layer_name, layer_data: np.ndarray): ...

    @abstractmethod
    def delete(self, layer_name): ...
    
    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def merge(
        self,
        layer_stack: Optional[LayerStackBase] = None,
        tiles_callback: Optional[Callable] = None,
    ):
        if layer_stack is None:
            return

        for layer in layer_stack:
            # Get a data layer for the corresponding kind
            kind = layer.kind
            data = layer.data
            meta = layer.meta
            name = layer.name

            layer_class = type(layer)
            layer_class.validate_data(data, meta)

            # Find an eventual existing layer, based on name
            existing_layer = self.read(name)

            algo_is_tiled = meta.get("tile_params") is not None

            if existing_layer is None:  # => Create
                # Resolve pixel domain
                if algo_is_tiled:
                    tile_params = meta.get("tile_params")
                    ndim = tile_params.get("ndim")
                    if ndim is not None:
                        pixel_domain = tuple(
                            [
                                tile_params.get(f"domain_size_{idx}")
                                for idx in range(ndim)
                            ]
                        )
                    else:
                        pixel_domain = layer_class.pixel_domain(data)
                else:
                    pixel_domain = layer_class.pixel_domain(data)

                # Initialize data
                initial_data = layer_class._get_initial_data(pixel_domain)

                # Create a new layer
                existing_layer = self.create(
                    kind, initial_data, name, meta
                )

            else:  # => Update
                # On first tile, initialize and update the existing layer
                if algo_is_tiled:
                    tile_params = meta.get("tile_params")
                    is_first_tile = tile_params.get("first_tile")
                    if is_first_tile is not None:
                        ndim = tile_params.get("ndim")
                        if ndim is not None:
                            pixel_domain = tuple(
                                [
                                    tile_params.get(f"domain_size_{idx}")
                                    for idx in range(ndim)
                                ]
                            )
                        else:
                            pixel_domain = layer.pixel_domain(data)

                        first_tile_data = layer._get_initial_data(pixel_domain)

                        # Update the existing layer
                        self.update(existing_layer.name, first_tile_data)

            # Refresh reference to the current data
            current_data = existing_layer.data

            # Resolving what the updated data should be
            if algo_is_tiled:
                # Add current_data to data
                updated_data = layer_class._merge_tile(current_data, data, meta)
            else:
                # Replace data by current_data
                updated_data = data

            # Updating the layer
            self.update(existing_layer.name, updated_data)

            if algo_is_tiled:
                # Emit a progress step
                if tiles_callback is not None:
                    tiles_callback(
                        tile_idx=meta.get("tile_params").get("tile_idx"),
                        n_tiles=meta.get("tile_params").get("n_tiles"),
                    )

    def samples_emitted(self, images):  # TODO: modify this
        for k, image in enumerate(images):
            self.create(
                kind="image",
                data=image,
                name=f"Sample image [{k}]",
            )


class Results(LayerStackBase):
    def __init__(self):
        super().__init__()
        self.layers = []

    def __repr__(self):
        return f"Results (Layers: {len(self.layers)})"

    def create(self, kind, data, name=None, meta=None):
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
        layer_class = DATA_TYPES.get(kind)
        if layer_class is None:
            raise ValueError(f"{layer_class} layers cannot be handled.")

        # Instanciate layer
        added_layer = layer_class(name=name, data=data, meta=meta)

        # Add layer to the stack
        self.layers.append(added_layer)

        return added_layer

    def read(self, layer_name):
        for layer in self.layers:
            if layer.name == layer_name:
                return layer

    def update(self, layer_name, updated_data: np.ndarray):
        layer = self.read(layer_name)
        layer.update(updated_data)
        return layer

    def delete(self, layer_name):
        for idx, layer in enumerate(self.layers):
            if layer.name == layer_name:
                self.layers.pop(idx)
