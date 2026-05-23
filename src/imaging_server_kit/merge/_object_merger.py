from typing import Dict, Union

import numpy as np

from imaging_server_kit.merge.layer_merger import DefaultMerger
from imaging_server_kit.types._points import Points
from imaging_server_kit.types._vectors import Vectors
from imaging_server_kit.types._boxes import Boxes
from imaging_server_kit.core.domain import merge_domains


def _merge_meta(incoming_meta_val, receiving_meta_val, n_objects_incoming):
    if isinstance(incoming_meta_val, Dict):
        return {
            k: _merge_meta(v, receiving_meta_val.get(k), n_objects_incoming)
            for k, v in incoming_meta_val.items()
        }
    if isinstance(incoming_meta_val, np.ndarray):
        if len(incoming_meta_val) == n_objects_incoming:
            if len(incoming_meta_val.shape) == 1:
                return np.hstack((receiving_meta_val, incoming_meta_val))
            else:
                return np.vstack((receiving_meta_val, incoming_meta_val))

    return incoming_meta_val


def merge_meta_tile(incoming_meta: Dict, receiving_meta: Dict, n_objects_incoming: int) -> Dict:
    return {
        k: _merge_meta(incoming_meta_val, receiving_meta.get(k), n_objects_incoming)
        for k, incoming_meta_val in incoming_meta.items()
    }


class ObjectMerger(DefaultMerger):
    @staticmethod
    def merge(
        receiving_layer: Union[Points, Vectors, Boxes],
        incoming_layer: Union[Points, Vectors, Boxes],
    ) -> None:
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return

        if incoming_layer.n_objects == 0:
            return

        if (receiving_layer.data is None) or (receiving_layer.position is None):
            receiving_layer.position = incoming_layer.position  # TODO: correct?
            receiving_layer.data = incoming_layer.data
            receiving_layer.meta = incoming_layer.meta
            return

        merged_extent = merge_domains(
            domains=[receiving_layer.extent, incoming_layer.extent]
        )
        
        new_position = merged_extent.coords_min
        
        new_receiving_layer_data = receiving_layer.data_from_coords(new_position)

        new_incoming_layer_data = incoming_layer.data_from_coords(new_position)

        new_data = np.vstack((new_receiving_layer_data, new_incoming_layer_data))

        receiving_layer.data = new_data

        receiving_layer.position = new_position

        new_meta = merge_meta_tile(
            incoming_layer.meta, receiving_layer.meta, incoming_layer.n_objects
        )

        receiving_layer.meta = new_meta