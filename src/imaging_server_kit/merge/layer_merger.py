from typing import Dict, List, Type

import numpy as np

from imaging_server_kit.types import DataLayer
from imaging_server_kit.merge.merger import Merger, DefaultMerger
from imaging_server_kit.merge._image_merger import (
    ImageOverrideMerger,
    ImageTileOverlapMerger,
)
from imaging_server_kit.merge._mask_merger import (
    InstanceMaskTileMerger,
    MaskOverrideMerger,
    MaskTileOverrideMerger,
)
from imaging_server_kit.merge._object_merger import ObjectMerger, ObjectTileMerger

from imaging_server_kit.core.domain import Domain


LAYER_MERGERS: Dict[str, Dict[str, Type[Merger]]] = {
    "image": {"default": ImageTileOverlapMerger, "override": ImageOverrideMerger},
    "mask": {
        "default": MaskTileOverrideMerger,
        "instances": InstanceMaskTileMerger,
        "override": MaskOverrideMerger,
    },
    "points": {"default": ObjectTileMerger, "override": ObjectMerger},
    "boxes": {"default": ObjectTileMerger, "override": ObjectMerger},
    "vectors": {"default": ObjectTileMerger, "override": ObjectMerger},
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
    def merge(receiving_layer: DataLayer, incoming_layer: DataLayer) -> None:
        if incoming_layer.tile_meta.is_first_tile:
            merger = find_layer_merger(receiving_layer)
            receiving_layer.merger_instance = merger
            merger.on_first_merge(receiving_layer, incoming_layer)
        else:
            merger = receiving_layer.merger_instance

        merger.merge(receiving_layer, incoming_layer)

        if incoming_layer.tile_meta.is_last_tile:
            merger.on_last_merge(receiving_layer, incoming_layer)


def merge_layers(layers: List[DataLayer]) -> DataLayer:
    """Merge a list of data layers of the same kind.
    Note: This method differs from layer.merge(other_layer) which is an in-place merge.
    Here, a new layer is created and the data from all `layers` are merged into it.
    """
    if len(layers) == 0:
        raise ValueError("There should be at least one layer to merge.")
    elif len(layers) == 1:
        return layers[0]

    first_layer = layers[0]
    cls = type(first_layer)

    # Check that the items in `layers` are all of the same type
    for l in layers[1:]:
        if not isinstance(l, cls):
            raise ValueError("Layers to merge must be of the same type.")

    # Find the layer bounds
    min_bounds = []
    max_bounds = []
    for l in layers:
        if l.coords_min is not None:
            min_bounds.append(l.coords_min)
        if l.coords_max is not None:
            max_bounds.append(l.coords_max)
    if len(min_bounds):
        min_bounds = np.min(np.stack(min_bounds), axis=0).tolist()
    if len(max_bounds):
        max_bounds = np.max(np.stack(max_bounds), axis=0).tolist()
        
    size = tuple(np.array(max_bounds) - np.array(min_bounds))
    domain = Domain(position=min_bounds, size=size)

    # Create a new instance
    merged_layer = cls(
        data=cls.initialize_data(domain=domain),
        name=first_layer.name,  # Use first layer name by convention
    )

    merger = LayerMerger()
    for l in layers:
        merger.merge(merged_layer, l)

    return merged_layer
