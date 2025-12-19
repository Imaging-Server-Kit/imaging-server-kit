from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from skimage.util import map_array

from imaging_server_kit.core.tiling import generate_nd_tiles
from ._mask import _get_slices, Mask


def unique_positive(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels[labels > 0])


def overlap_border_mask(tile_info: Dict, tile_shape: Tuple) -> np.ndarray:
    """Returns a boolean array selecting the rectangular region overalpping with other tiles."""
    overlaps_px = tuple(
        [
            tile_info["tile_params"][f"overlap_px_{idx}"]
            for idx in range(tile_info["tile_params"]["ndim"])
        ]
    )
    overlap_slices = tuple(
        [slice(pos, max_pos - pos) for pos, max_pos in zip(overlaps_px, tile_shape)]
    )
    mask = np.ones(tile_shape)
    mask[overlap_slices] = 0
    return mask == 1  # Make it bool type


class InstanceTileTracker:
    def __init__(self) -> None:
        self.N = 0  # Current number of objects
        self.G = nx.Graph()

    def add_N_to_tile(self, labels: np.ndarray) -> np.ndarray:
        labels[labels != 0] = labels[labels != 0] + self.N
        self.N = labels.max()
        return labels

    def add_node(self, lab):
        self.G.add_node(lab)

    def add_edge(self, a, b):
        self.G.add_edge(a, b)

    def build_mapping(self):
        self.G.add_nodes_from(range(1, self.N + 1))
        mapping = {}
        for comp_id, comp in enumerate(nx.connected_components(self.G), start=1):
            for n in comp:
                mapping[n] = comp_id
        self._mapping = mapping

    def resolve(self, arr):
        if not hasattr(self, "_mapping"):
            self.build_mapping()
        # map_array wants parallel arrays:
        input_vals = np.array(list(self._mapping.keys()), dtype=np.int64)
        output_vals = np.array([self._mapping[k] for k in input_vals], dtype=np.int64)
        return map_array(arr, input_vals=input_vals, output_vals=output_vals)


class InstanceMask(Mask):
    """Data layer used to represent instance segmentation masks.

    Parameters
    ----------
    data: Numpy arrays, integer type. Integers can represent object classes (e.g. pixel classification) or object instances.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    min_intersecting_px: Minimum number of intersecting pixels between overlapping tile instances to merge individual objects.
    """

    kind = "instance_mask"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name: str = "Mask",
        description: str = "Segmentation mask (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        required: bool = True,
        meta: Optional[Dict] = None,
        min_intersecting_px: int=5,
    ):
        super().__init__(
            data=data,
            name=name,
            description=description,
            dimensionality=dimensionality,
            required=required,
            meta=meta,
        )
        self.tile_tracker = InstanceTileTracker()
        self.min_intersecting_px = min_intersecting_px
    
    def merge_tile(self, mask_tile: np.ndarray, tile_info: Dict) -> None:
        if self.data is None:
            return
        
        if tile_info["tile_params"].get("first_tile") is True:
            self.tile_tracker = InstanceTileTracker()
        
        tile_slices = _get_slices(self.data, tile_info)
        src_arr = self.tile_tracker.add_N_to_tile(mask_tile)
        dst_arr = self.data[tile_slices].copy()
        self.data[tile_slices] = src_arr
        
        for new_label in unique_positive(src_arr):
            self.tile_tracker.add_node(new_label)
        
        border_mask = overlap_border_mask(tile_info, src_arr.shape)
        for src_lab in unique_positive(src_arr[border_mask]):
            lab_filt_src_in_overlap = np.logical_and(border_mask, src_arr == src_lab)
            lab_filt_dst_in_overlap = dst_arr[lab_filt_src_in_overlap]
            for dst_lab in unique_positive(lab_filt_dst_in_overlap):
                n_intersecting_px = (lab_filt_dst_in_overlap == dst_lab).sum()
                if n_intersecting_px > self.min_intersecting_px:
                    self.tile_tracker.add_edge(dst_lab, src_lab)
        
        # Do the graph merging at the last iteration
        # Progress might freeze at 100% for a while, but that's OK for now..
        is_last_tile = tile_info["tile_params"]["tile_idx"] == tile_info["tile_params"]["n_tiles"]-1
        if is_last_tile:
            pixel_domain = self.pixel_domain()
            instance_mask = Mask(self._get_initial_data(pixel_domain))
            for tile_info in generate_nd_tiles(pixel_domain, tile_size_px=512):
                mask_tile, _ = self.get_tile(tile_info)
                resolved_mask_tile = self.tile_tracker.resolve(mask_tile)
                instance_mask.merge_tile(resolved_mask_tile, tile_info)
            
            # Update the data
            self.data = instance_mask.data