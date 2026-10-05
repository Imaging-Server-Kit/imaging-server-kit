from typing import Dict, List, Type

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types import Layer, layer_factory
from imaging_server_kit.types.common import copy_meta
from imaging_server_kit.merge.merger import Merger, DefaultMerger
from imaging_server_kit.merge._image_merger import ImageTileOverlapMerger
from imaging_server_kit.merge._mask_merger import (
    InstanceMaskTileMerger,
    MaskOverrideMerger,
)
from imaging_server_kit.merge._object_merger import ObjectMerger

LAYER_MERGERS: Dict[str, Dict[str, Type[Merger]]] = {
    "image": {"default": ImageTileOverlapMerger},
    "mask": {
        "default": MaskOverrideMerger,
        "instances": InstanceMaskTileMerger,
    },
    "points": {"default": ObjectMerger},
    "boxes": {"default": ObjectMerger},
    "vectors": {"default": ObjectMerger},
}


def find_layer_merger(layer: Layer) -> Merger:
    if layer.kind in LAYER_MERGERS:
        lm = LAYER_MERGERS[layer.kind]
        merger_cls = lm.get(layer.meta["merger"], DefaultMerger)
    else:
        merger_cls = DefaultMerger

    return merger_cls()


class LayerMerger:
    """Dispatches layer merging to the strategy matching the layer kind and `meta["merger"]`.

    Used internally by `Stack.merge()` to merge the result layers of one tile into an
    accumulating stack, and by `merge_layers()` to merge a list of layers.
    """

    @staticmethod
    def merge(
        receiving_layer: Layer, incoming_layer: Layer, merge_data: bool = True
    ) -> None:
        """Merge `incoming_layer` into `receiving_layer`, in place.

        Parameters
        ----------
        receiving_layer : Layer
            The layer to merge into. Modified in place.
        incoming_layer : Layer
            The layer being merged in.
        merge_data : bool, default=True
            Whether to merge the data of the layers. If `False`, only the first- and
            last-tile hooks of the merger (`on_first_merge`, `on_last_merge`) are run.
        """
        if incoming_layer.tile_meta.is_first_tile:
            merger = find_layer_merger(receiving_layer)
            receiving_layer._merger_instance = merger
            merger.on_first_merge(receiving_layer, incoming_layer)
        else:
            merger = receiving_layer._merger_instance
            if merger is None:
                # Make sure to have at least a DefaultMerger instance:
                merger = find_layer_merger(receiving_layer)

        if merge_data:
            merger.merge(receiving_layer, incoming_layer)

        if incoming_layer.tile_meta.is_last_tile:
            merger.on_last_merge(receiving_layer, incoming_layer)


def merge_layers(layers: List[Layer]) -> Layer:
    """Merge a list of data layers of the same kind into a new layer.

    The layers are merged as successive tiles, following the merging strategy of the
    layer type (see the "Tile merging" section of the documentation). The input layers
    are not modified.

    Parameters
    ----------
    layers : list of Layer
        The layers to merge. They must all be of the same kind.

    Returns
    -------
    Layer
        A new layer containing the merged data. If a single layer is given, it is
        returned as is.

    Raises
    ------
    ValueError
        If `layers` is empty, or if the layers are not all of the same kind.
    """
    if len(layers) == 0:
        raise ValueError("There should be at least one layer to merge.")
    elif len(layers) == 1:
        return layers[0]

    first_layer = layers[0]
    kind = first_layer.kind
    name = first_layer.name
    meta = first_layer.meta

    for l in layers[1:]:
        if l.kind != kind:
            raise ValueError("Layers to merge must be of the same kind.")

    merged_layer = layer_factory(kind=kind, name=name, **meta)

    # The layers are merged as a series of tiles (first => last) of a single merge run,
    # so that e.g. instance labels from different layers don't collide.
    # We merge copies, not to modify the tile metadata of the provided layers.
    n_layers = len(layers)
    merger = LayerMerger()
    for idx, l in enumerate(layers):
        incoming_layer = type(l)(
            data=l.data,
            name=l.name,
            meta=copy_meta(l),
            tile_meta=TileMeta(tile_idx=idx, n_tiles=n_layers),
        )
        merger.merge(merged_layer, incoming_layer)

    return merged_layer
