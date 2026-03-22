"""Common utilities for data layers (merging and accessing metadata in tiles, for Points, Vectors, Boxes, etc.)"""

from typing import Dict
import numpy as np


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
