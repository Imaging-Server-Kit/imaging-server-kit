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
        if (incoming_layer.data is None) or (incoming_layer.ndim is None):
            return
        
        if incoming_layer.n_objects == 0:
            return

        if (receiving_layer.data is None) or (receiving_layer.ndim is None):
            receiving_layer.data = incoming_layer.initialize_data(incoming_layer.domain)
            receiving_layer.domain.coords_min = incoming_layer.domain.coords_min
            receiving_layer.domain.size = incoming_layer.domain.size
            receiving_layer.meta = incoming_layer.meta

        cmin_diff = []
        cmin_inc = []
        xxx = []
        for receiving_cmin, incoming_cmin in zip(
            receiving_layer.coords_min,
            incoming_layer.coords_min,
        ):
            diff = incoming_cmin - receiving_cmin
            start_receiving = -diff if diff < 0 else 0
            start_incoming = 0 if diff < 0 else -diff
            cmin_diff.append(start_receiving)
            cmin_inc.append(start_incoming)
            xxx.append(diff)

        receiving_layer.domain.coords_min = tuple(
            np.array(receiving_layer.domain.coords_min) - np.array(cmin_diff)
        )
        
        inc_data = incoming_layer.data.copy()

        for dim in range(incoming_layer.ndim):
            if incoming_layer.kind == "points":
                inc_data[:, dim] = inc_data[:, dim] - np.array(cmin_inc)[dim]
            elif incoming_layer.kind == "boxes":
                inc_data[:, :, dim] = inc_data[:, :, dim] - np.array(cmin_inc)[dim]
            elif incoming_layer.kind == "vectors":
                inc_data[:, 0, dim] = inc_data[:, 0, dim] - np.array(cmin_inc)[dim]
        
        if receiving_layer.n_objects > 0:
            
            for dim in range(receiving_layer.ndim):
                if receiving_layer.kind == "points":
                    receiving_layer.data[:, dim] = receiving_layer.data[:, dim] + np.array(cmin_diff)[dim]
                elif receiving_layer.kind == "boxes":
                    receiving_layer.data[:, :, dim] = receiving_layer.data[:, :, dim] + np.array(cmin_diff)[dim]
                elif receiving_layer.kind == "vectors":
                    receiving_layer.data[:, 0, dim] = receiving_layer.data[:, 0, dim] + np.array(cmin_diff)[dim]
        
            new_data = np.vstack(
                (
                    receiving_layer.data,
                    inc_data,
                )
            )

            new_meta = merge_meta_tile(
                receiving_layer.meta, incoming_layer.meta, receiving_layer.n_objects
            )
            
        else:
            new_data = incoming_layer.data
            new_meta = incoming_layer.meta

        receiving_layer.data = new_data
        receiving_layer.meta = new_meta

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
        receiving_layer.data = receiving_layer.initialize_data(incoming_layer.domain)

        receiving_layer.domain.coords_min = incoming_layer.domain.coords_min
        receiving_layer.domain.size = incoming_layer.domain.size

        receiving_layer.meta = incoming_layer.meta
