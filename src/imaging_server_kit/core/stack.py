from __future__ import annotations

from typing import Generator, List, Optional, Tuple, Union

from imaging_server_kit.merge.layer_merger import LayerMerger
from imaging_server_kit.types import Layer
from imaging_server_kit.core.domain import Domain, merge_domains
from imaging_server_kit.core.tiling import (
    TileMeta,
    TilingSpecs,
    generate_tiles,
)


class Stack:
    """A stack of data layers.

    Access layers by index: `layer = stack[0]` or name: `layer = stack.read("Layer Name")`.

    Attributes
    ----------
    layers: List of layers in the stack.
    tile_meta: Tile metadata for the stack.
    coords_min: Minimum coordinates of the stack, inferred from the layers.
    coords_max: Maximum coordinates of the stack, inferred from the layers.
    domain: Domain of the stack, inferred from coords_min and coords_max.
    size: Size of the stack's domain.
    ndim: Dimensionality of the stack, inferred from the domain.
    position: Position of the stack in global pixel space.

    Methods
    ----------
    add(): Add a layer to the stack.
    read(): Read a layer by name.
    delete(): Delete a layer by name.
    merge(): Merge another layer stack.
        Layers of the same kind, with the same name will be updated (meta and data). Other layers will be added to the stack.
    select(): Select stack data in a given domain.
    """

    def __init__(
        self,
        layers: Optional[List[Layer]] = None,
        tile_meta: Optional[TileMeta] = None,
        position: Optional[Tuple] = None,
    ):
        self._layers: List[Layer] = []
        if layers is not None:
            for layer in layers:
                self.add(layer)

        self._tile_meta = TileMeta() if tile_meta is None else tile_meta.copy()
        
        self._position = position

    def __str__(self):
        message = f"Stack (Layers: {len(self.layers)})"
        for l in self.layers:
            message += "\n"
            message += l.__str__()
        return message

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, key) -> List[Layer]:
        # Stacks have a `layers` dimension (first dimension)
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
                        size.append(self.size[dim])

            domain = Domain(size=size, position=position)

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

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @property
    def tile_meta(self) -> TileMeta:
        return self._tile_meta

    @tile_meta.setter
    def tile_meta(self, value: TileMeta):
        self._tile_meta = value
        
        # Setting the tile_meta of the stack sets the tile metas of all layers
        for l in self.layers:
            l.tile_meta = value

    @property
    def extent(self) -> Optional[Domain]:
        return merge_domains(domains=[l.extent for l in self.layers])

    @property
    def ndim(self) -> Optional[int]:
        if self.extent is not None:
            return self.extent.ndim

    @property
    def size(self) -> Optional[Union[Tuple, List]]:
        if self.extent is not None:
            return self.extent.size

    @property
    def coords_min(self) -> Optional[Tuple]:
        return self.extent.coords_min

    @property
    def coords_max(self) -> Optional[Tuple]:
        return self.extent.coords_max
    
    @property
    def position(self) -> Optional[Tuple]:
        if self._position is not None:
            return self._position

    @position.setter
    def position(self, value):
        self._position = value
        
        # Setting the position of the stack sets the positions of all layers
        for l in self.layers:
            l.position = value

    def add(self, layer: Layer) -> Layer:
        """Add a new layer to the layer stack.

        Parameters
        ----------
        layer: Layer to add to the stack. If a layer with that name already exists, its name will be changed with a suffix (e.g. Image-01).
        """
        new_name = self._resolve_layer_name(layer.kind, layer.name)
        if new_name != layer.name:
            layer.name = new_name
        self._layers.append(layer)
        
        # Trigger the `post_add` event:
        self.post_add(layer)
        
        return layer

    def post_add(self, layer: Layer):
        pass

    def merge(
        self,
        stack: Optional[Stack],
        reinitialize_domain: Optional[Domain] = None,
    ) -> None:
        """Merge another layer stack.

        Notes
        ----------
        - Layers with the same name are merged, while layers with a different name are added.
        - Ends by triggering a `post_merge` event (empty by default).

        Parameters
        ----------
        stack: Layer stack to be merged.
            Layers from this stack of the same kind, with the same name as instance layers (self.layers)
            will update corresponding meta and data attributes.
            Other layers from layer_stack will be added via the create() method.
        reinitialize_domain: Optional domain to reinitialize before merging the incoming stack.
        """
        if stack is None:
            return

        to_merge = []
        receiving_layers = []
        for incoming_layer in stack:
            receiving_layer = self.read(incoming_layer.name)
            to_merge.append(receiving_layer is not None)
            if receiving_layer is None:
                # We add the incoming layer and do not merge data (itself) into it:
                receiving_layer = self.add(incoming_layer)
            else:
                # First tiles always reinitialize the domain:
                if (incoming_layer.tile_meta.is_first_tile) & isinstance(
                    reinitialize_domain, Domain
                ):
                    receiving_layer.reinitialize(reinitialize_domain)

            receiving_layers.append(receiving_layer)

        layer_merger = LayerMerger()
        for receiving_layer, incoming_layer, merge_data in zip(
            receiving_layers, stack, to_merge
        ):
            layer_merger.merge(receiving_layer, incoming_layer, merge_data=merge_data)

        self.post_merge(receiving_layers)

    def post_merge(self, receiving_layers: List[Layer]):
        pass

    def delete(self, name: str) -> None:
        """Delete a layer by name."""
        for idx, layer in enumerate(self.layers):
            if layer.name == name:
                self._layers.pop(idx)

        self.post_delete(name)

    def post_delete(self, name: str) -> None:
        pass

    def read(self, name: str) -> Optional[Layer]:
        """Read a layer by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer

    def select(self, domain: Domain) -> Stack:
        """Selet a sub-stack in the given domain."""
        return Stack(layers=[l.select(domain=domain.copy()) for l in self.layers])

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


class StackTileGenerator:
    @staticmethod
    def generate_tiles(
        stack: Stack, ctx: Optional[TilingSpecs]
    ) -> Generator[Stack, None, None]:
        if ctx is None:
            yield stack.select(domain=Domain())
        else:
            for tile_meta, tile_domain in generate_tiles(
                domain=stack.extent,
                tile_size=ctx.tile_size,
                tile_overlap=ctx.tile_overlap,
                tile_delay=ctx.tile_delay,
                tile_order_random=ctx.tile_order_random,
            ):
                stack_tile = stack.select(domain=tile_domain)
                stack_tile.tile_meta = tile_meta
                stack_tile.position = tile_domain.coords_min
                
                yield stack_tile
