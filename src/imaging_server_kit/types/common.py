"""Common utilities for data layers (merging and accessing metadata in tiles, for Points, Vectors, Boxes, etc.)"""

from typing import Dict, Optional
import numpy as np

from imaging_server_kit.types.data_layer import DefaultMerger, DataLayer


def _extract_meta(obj, n_objects, tile_filter):
    if isinstance(obj, Dict):
        return {k: _extract_meta(v, n_objects, tile_filter) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if len(obj) == n_objects:
            return obj[tile_filter]
    return obj


def extract_meta_tile(meta: Dict, n_objects: int, tile_filter: np.ndarray) -> Dict:
    """Iterates over two levels of the meta dictionary.
    Any numpy array found with length==n_objects is filtered using tile_filter.
    """
    assert len(tile_filter) == n_objects
    return {k: _extract_meta(v, n_objects, tile_filter) for k, v in meta.items()}


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


class ObjectMerger(DefaultMerger):
    def merge(self, src_layer: DataLayer, dst_layer: DataLayer) -> None:
        if (
            (dst_layer.data is None)
            or (dst_layer.tile_meta is None)
            or (dst_layer.pixel_domain is None)
        ):
            return

        if (
            (src_layer.data is None)
            or (src_layer.tile_meta is None)
            or (src_layer.pixel_domain is None)
        ):
            src_layer.data = dst_layer.data_global_coords
            src_layer.meta = dst_layer.meta
        else:
            if src_layer.n_objects > 0:
                merged_data = np.vstack(
                    (src_layer.data_global_coords, dst_layer.data_global_coords)
                )
                merged_data = merged_data - src_layer.tile_meta.coords_min
                merged_meta = merge_meta_tile(
                    src_layer.meta, dst_layer.meta, src_layer.n_objects
                )
            else:
                merged_data = dst_layer.data
                merged_meta = dst_layer.meta

            src_layer.data = merged_data
            src_layer.meta = merged_meta


class ObjectTileMerger(ObjectMerger):
    def first_tile_hook(self, src_layer: DataLayer, dst_layer: DataLayer):
        # Erase all of the data before tiling
        src_layer.data = src_layer._get_initial_data(src_layer.data_pixel_domain)
        # src_layer.meta = dst_layer.meta  # TODO: Needed?