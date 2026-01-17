from typing import Dict, List, Optional

import numpy as np
import networkx as nx
from skimage.util import map_array

from imaging_server_kit.core.tiling import TileMeta

from ._mask import Mask


def unique_positive(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels[labels > 0])


class InstanceTileTracker:
    def __init__(self) -> None:
        self.initialize()

    def initialize(self):
        self.N = 0  # Current number of objects
        self.G = nx.Graph()

    def add_N_to_tile(self, labels: np.ndarray) -> np.ndarray:
        if labels.sum() > 0:
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
        tile_meta: Optional[TileMeta] = None,
        min_intersecting_px: int = 5,
    ):
        super().__init__(
            data=data,
            name=name,
            description=description,
            dimensionality=dimensionality,
            required=required,
            meta=meta,
            tile_meta=tile_meta,
        )
        self.tile_tracker = InstanceTileTracker()
        self.min_intersecting_px = min_intersecting_px

    def first_tile_hook(self):
        self.tile_tracker.initialize()

    def last_tile_hook(self):
        # Do the graph merging (Note: could be done tile-by-tile as well)
        self.data = self.tile_tracker.resolve(self.data)

    def merge(self, mask_tile: Mask) -> None:
        if (self.data is None) or (mask_tile.tile_meta is None):
            raise RuntimeError("Invalid attempt to merge an instance mask tile.")

        dst_tile = self.get_tile(mask_tile.tile_meta)
        if dst_tile is None:
            raise RuntimeError("Invalid attempt to merge an instance mask tile.")

        src_arr = self.tile_tracker.add_N_to_tile(mask_tile.data)

        self.data[mask_tile.tile_meta.slices] = src_arr

        for new_label in unique_positive(src_arr):
            self.tile_tracker.add_node(new_label)

        border_mask = mask_tile.tile_meta.overlap_border_mask

        for src_lab in unique_positive(src_arr[border_mask]):
            lab_filt_src_in_overlap = np.logical_and(border_mask, src_arr == src_lab)
            lab_filt_dst_in_overlap = dst_tile.data[lab_filt_src_in_overlap]
            for dst_lab in unique_positive(lab_filt_dst_in_overlap):
                n_intersecting_px = (lab_filt_dst_in_overlap == dst_lab).sum()
                if n_intersecting_px > self.min_intersecting_px:
                    self.tile_tracker.add_edge(dst_lab, src_lab)