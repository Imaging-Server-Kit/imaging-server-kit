from typing import Dict, List, Type

from imaging_server_kit.types import Layer, layer_factory
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
    """Dispatches layer merging to the appropriate `Merger` strategy based on layer kind and `meta["merger"]`.

    Used internally by `Stack.merge()` to merge one tile's result layers into an
    accumulating stack; also used by `merge_layers()` to merge a list of layers directly.

    Methods
    ----------
    merge(): Merge an incoming layer into a receiving layer, in place.
    """

    @staticmethod
    def merge(
        receiving_layer: Layer, incoming_layer: Layer, merge_data: bool = True
    ) -> None:
        """Merge `incoming_layer` into `receiving_layer`, in place.

        Parameters
        ----------
        receiving_layer: The layer to merge into. Modified in place.
        incoming_layer: The layer being merged in.
        merge_data: Whether to merge the layers' data. If False, only the first/last-tile
            merger lifecycle hooks (`on_first_merge`/`on_last_merge`) are run.
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

    Note: unlike `LayerMerger.merge()`, which merges in place into an existing layer,
    this creates a new layer and merges the data from all `layers` into it.

    Parameters
    ----------
    layers: Layers to merge. Must all be of the same kind.

    Returns
    -------
    A new layer containing the merged data.
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

    merger = LayerMerger()
    for l in layers:
        merger.merge(merged_layer, l)

    return merged_layer
