from __future__ import annotations

from typing import Generator, List, Optional, Tuple, Union

from imaging_server_kit.merge.layer_merger import LayerMerger
from imaging_server_kit.types import Layer
from imaging_server_kit.core.domain import Domain, domain_from_key, index_frame, merge_domains
from imaging_server_kit.core._fmt import fmt_slices
from imaging_server_kit.core.tiling import (
    TileMeta,
    TilingSpecs,
    generate_tiles,
)


class Stack:
    """An ordered collection of data layers.

    Access layers by index (`stack[0]`) or by name (`stack.read("Layer name")`).
    Further indices are spatial, as with NumPy arrays: `stack[:, 50:150]` selects rows
    50 to 150 of all layers.

    Parameters
    ----------
    layers : list of Layer, optional
        Initial layers of the stack.
    tile_meta : TileMeta, optional
        Metadata about the stack's position and role in a set of tiles.
    position : tuple, optional
        Position of the stack in global pixel coordinates.

    Attributes
    ----------
    layers : list of Layer
        The layers in the stack.
    position : tuple
        Position of the stack in global pixel coordinates. Setting it offsets the
        positions of all layers.
    extent : Domain
        Smallest region containing all layers, in global pixel coordinates.
    size : tuple
        Size of the extent.
    coords_min : tuple
        Lower corner of the extent.
    coords_max : tuple
        Upper corner of the extent.
    ndim : int
        Number of spatial dimensions of the extent.
    tile_meta : TileMeta
        Metadata about the stack's position and role in a set of tiles.
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

        self._position = position  # self._resolve_stack_position(position)

    def __repr__(self) -> str:
        n = len(self.layers)
        message = f"<Stack {n} layer{'' if n == 1 else 's'}"
        if self.extent is not None:
            message += f", extent {fmt_slices(self.coords_min, self.coords_max)}"
        return message + ">"

    def __str__(self) -> str:
        if len(self.layers) == 0:
            return "Stack | empty"

        header = [f"Stack | {len(self.layers)} layers"]
        if self.extent is not None:
            header.append(f"extent {fmt_slices(self.coords_min, self.coords_max)}")
        if self.tile_meta.n_tiles > 1:
            header.append(f"tile {self.tile_meta.tile_idx}/{self.tile_meta.n_tiles}")

        rows = [("#", "kind", "name", "data")]
        rows += [(str(i), l.kind, l.name, l._summary()) for i, l in enumerate(self.layers)]
        # Only show the first and last layers of very large stacks
        if len(rows) > 21:
            rows = rows[:11] + [("...", "", "", "")] + rows[-5:]

        widths = [max(len(row[col]) for row in rows) for col in range(3)]
        lines = [" | ".join(header)]
        for row in rows:
            cells = [cell.ljust(width) for cell, width in zip(row, widths)]
            lines.append(f"  {'  '.join(cells)}  {row[3]}".rstrip())
        return "\n".join(lines)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, key) -> Union[Layer, List[Layer]]:
        # Stacks have a `layers` dimension (first dimension)
        # so we index as [Layer, Dim0, Dim1, .., DimN]
        if not isinstance(key, tuple):
            key = (key,)

        layer_key = key[0]
        if layer_key is Ellipsis:
            raise IndexError("Ellipsis ('...') is not supported for the `layers` dimension")

        if len(key) > 1:
            # Spatial indices are counted from the smallest layer position
            positions = [l.position for l in self.layers if l.position is not None]
            origin = tuple(map(min, zip(*positions))) if positions else None
            frame = index_frame(origin, self.extent)
            extract = self.select(domain=domain_from_key(key[1:], frame))
        else:
            extract = self

        # The first key indexes the `layers` dimension (int or slice)
        return extract.layers[layer_key]

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
        if self.extent is not None:
            return self.extent.coords_min

    @property
    def coords_max(self) -> Optional[Tuple]:
        if self.extent is not None:
            return self.extent.coords_max

    @property
    def position(self) -> Optional[Tuple]:
        if self._position is not None:
            return self._position

        # If any of the layers has a position set, we set the stack's position to zero
        for l in self.layers:
            if l.position is not None:
                ndim = len(l.position)
                return tuple([0] * ndim)

    @position.setter
    def position(self, value: Optional[Tuple]):
        self._position = value

        # Setting the position of the stack can set or offset the positions of the layers
        if value is not None:
            for l in self.layers:
                if l.position is None:
                    l.position = value
                else:
                    l.position = tuple([p + q for p, q in zip(l.position, value)])

    def add(self, layer: Layer) -> Layer:
        """Add a layer to the stack.

        Parameters
        ----------
        layer : Layer
            The layer to add. If a layer with the same name already exists, a suffix
            is appended to the name of the new layer (e.g. `Image-01`).

        Returns
        -------
        Layer
            The added layer.
        """
        new_name = self._resolve_layer_name(layer.kind, layer.name)
        if new_name != layer.name:
            layer.name = new_name
        self._layers.append(layer)

        self._post_add(layer)

        return layer

    def _post_add(self, layer: Layer):
        """Event triggered after a layer is added to the stack."""
        pass

    def merge(
        self,
        stack: Optional[Stack],
        reinitialize_domain: Optional[Domain] = None,
    ) -> None:
        """Merge another stack into this one, in place.

        Incoming layers with the same name as an existing layer are merged into that
        layer, following the merging strategy of the layer type. Other incoming layers
        are added to the stack.

        Parameters
        ----------
        stack : Stack, optional
            The stack to merge. If `None`, nothing happens.
        reinitialize_domain : Domain, optional
            A region of the existing layers to reinitialize before merging the first
            tile of the incoming stack.
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
                # First tiles reinitialize the domain:
                if (incoming_layer.tile_meta.is_first_tile) and isinstance(
                    reinitialize_domain, Domain
                ):
                    receiving_layer._reinitialize(reinitialize_domain)

            receiving_layers.append(receiving_layer)

        layer_merger = LayerMerger()
        for receiving_layer, incoming_layer, merge_data in zip(
            receiving_layers, stack, to_merge
        ):
            layer_merger.merge(receiving_layer, incoming_layer, merge_data=merge_data)

        self._post_merge(receiving_layers)

    def _post_merge(self, receiving_layers: List[Layer]):
        """Event triggered after a layer is merged into the stack."""
        # Layers such as notifications and progress bars are shown in the terminal here,
        # once per merged result (subclasses, e.g. for Napari, display them differently).
        for layer in receiving_layers:
            layer._display()

    def delete(self, name: str) -> None:
        """Delete a layer by name.

        Parameters
        ----------
        name : str
            Name of the layer to delete.
        """
        for idx, layer in enumerate(self.layers):
            if layer.name == name:
                self._layers.pop(idx)

        self._post_delete(name)

    def _post_delete(self, name: str) -> None:
        """Event triggered after a layer is removed from the stack."""
        pass

    def read(self, name: str) -> Optional[Layer]:
        """Get a layer by name.

        Parameters
        ----------
        name : str
            Name of the layer.

        Returns
        -------
        Layer or None
            The layer, or `None` if there is no layer with that name.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer

    def select(self, domain: Domain) -> Stack:
        """Select the part of the stack inside a domain.

        Parameters
        ----------
        domain : Domain
            The region to select, in global pixel coordinates.

        Returns
        -------
        Stack
            A new stack with the selected part of each layer.
        """
        return Stack(layers=[l.select(domain=domain._copy()) for l in self.layers])

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
                tile_randomize=ctx.tile_randomize,
            ):
                stack_tile = stack.select(domain=tile_domain)
                stack_tile.tile_meta = tile_meta
                
                # We assign the tile position to the stack as well
                stack_tile.position = tile_domain.coords_min

                yield stack_tile
