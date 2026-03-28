from typing import Dict, Optional, Union

import numpy as np

from imaging_server_kit.merge.layer_merger import Merger
from imaging_server_kit.types._points import Points
from imaging_server_kit.types._vectors import Vectors
from imaging_server_kit.types._boxes import Boxes


def _merge_meta(obj, meta_obj, n_objects, tile_filter: Optional[np.ndarray]):
    if isinstance(obj, Dict):
        return {
            k: _merge_meta(v, meta_obj.get(k), n_objects, tile_filter)
            for k, v in obj.items()
        }
    if isinstance(obj, np.ndarray):
        if len(obj) == n_objects:
            if len(meta_obj.shape) == 1:
                if tile_filter:
                    return np.hstack((obj[~tile_filter], meta_obj))
                else:
                    return np.hstack((obj, meta_obj))
            else:
                if tile_filter:
                    return np.vstack((obj[~tile_filter], meta_obj))
                else:
                    return np.vstack((obj, meta_obj))
    return obj


def merge_meta_tile(
    meta: Dict,
    meta_tile: Dict,
    n_objects: int,
    tile_filter: Optional[np.ndarray] = None,
) -> Dict:
    """Iterates over two levels of meta.
    Merge meta from the tile with the existing meta for numpy array fields of length == n_objects.
    """
    return {
        k: _merge_meta(v, meta_tile.get(k), n_objects, tile_filter)
        for k, v in meta.items()
    }


class ObjectMerger(Merger):
    @staticmethod
    def merge(
        receiving_layer: Union[Points, Vectors, Boxes],
        incoming_layer: Union[Points, Vectors, Boxes],
    ) -> None:
        if (incoming_layer.data is None) or (incoming_layer.coords_max is None):
            return

        if (receiving_layer.data is None) or (receiving_layer.coords_max is None):
            receiving_layer.data = incoming_layer.data_global_coords
            receiving_layer.meta = incoming_layer.meta
        else:
            if receiving_layer.n_objects > 0:
                merged_data = np.vstack(
                    (
                        receiving_layer.data_global_coords,
                        incoming_layer.data_global_coords,
                    )
                )
                merged_data = merged_data - receiving_layer.domain.coords_min
                merged_meta = merge_meta_tile(
                    receiving_layer.meta, incoming_layer.meta, receiving_layer.n_objects
                )
            else:
                merged_data = incoming_layer.data
                merged_meta = incoming_layer.meta

            receiving_layer.data = merged_data
            receiving_layer.meta = merged_meta

    @staticmethod
    def on_first_merge(
        receiving_layer: Union[Points, Vectors, Boxes],
        incoming_layer: Union[Points, Vectors, Boxes],
    ):
        pass

    @staticmethod
    def on_last_merge(
        receiving_layer: Union[Points, Vectors, Boxes],
        incoming_layer: Union[Points, Vectors, Boxes],
    ):
        pass


class ObjectTileMerger(ObjectMerger):
    def on_first_merge(
        self,
        receiving_layer: Union[Points, Vectors, Boxes],
        incoming_layer: Union[Points, Vectors, Boxes],
    ):
        # Erase all of the data before tiling
        receiving_layer.data = receiving_layer.initialize_data(
            receiving_layer._data_bounds
        )
        # receiving_layer.meta = incoming_layer.meta  # TODO: Needed?
