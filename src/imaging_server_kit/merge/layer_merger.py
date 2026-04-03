from typing import Dict, List, Type

from imaging_server_kit.types import DataLayer, layer_factory
from imaging_server_kit.merge.merger import Merger, DefaultMerger
from imaging_server_kit.merge._image_merger import (
    ImageOverrideMerger,
    ImageTileOverlapMerger,
)
from imaging_server_kit.merge._mask_merger import (
    InstanceMaskTileMerger,
    MaskOverrideMerger,
)
from imaging_server_kit.merge._object_merger import ObjectMerger


LAYER_MERGERS: Dict[str, Dict[str, Type[Merger]]] = {
    "image": {"default": ImageTileOverlapMerger, "override": ImageOverrideMerger},
    "mask": {
        "default": MaskOverrideMerger,
        "instances": InstanceMaskTileMerger,
    },
    "points": {"default": ObjectMerger},
    "boxes": {"default": ObjectMerger},
    "vectors": {"default": ObjectMerger},
}


def find_layer_merger(layer: DataLayer) -> Merger:
    if layer.kind in LAYER_MERGERS:
        lm = LAYER_MERGERS[layer.kind]
        merger_cls = lm.get(layer.merger, DefaultMerger)
    else:
        merger_cls = DefaultMerger

    return merger_cls()


class LayerMerger:
    @staticmethod
    def merge(
        receiving_layer: DataLayer, incoming_layer: DataLayer, merge_data: bool = True
    ) -> None:
        if incoming_layer.tile_meta.is_first_tile:
            merger = find_layer_merger(receiving_layer)
            receiving_layer.merger_instance = merger
            merger.on_first_merge(receiving_layer, incoming_layer)
        else:
            merger = receiving_layer.merger_instance

        if merge_data:
            merger.merge(receiving_layer, incoming_layer)

        if incoming_layer.tile_meta.is_last_tile:
            merger.on_last_merge(receiving_layer, incoming_layer)


def merge_layers(layers: List[DataLayer]) -> DataLayer:
    """Merge a list of data layers of the same kind.
    Note: This method differs from layer.merge(other_layer), which is an in-place merge.
    Here, a new layer is created and the data from all `layers` are merged into it.
    """
    if len(layers) == 0:
        raise ValueError("There should be at least one layer to merge.")
    elif len(layers) == 1:
        return layers[0]

    first_layer = layers[0]
    kind = first_layer.kind
    name = first_layer.name

    for l in layers[1:]:
        if l.kind != kind:
            raise ValueError("Layers to merge must be of the same kind.")

    merged_layer = layer_factory(kind=kind, name=name)

    merger = LayerMerger()
    for l in layers:
        merger.merge(merged_layer, l)

    return merged_layer
