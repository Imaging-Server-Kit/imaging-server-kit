from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.common import (
    extract_meta_tile,
    ObjectMerger,
    ObjectTileMerger,
)
from imaging_server_kit.types.data_layer import DataLayer, Merger


class Boxes(DataLayer):
    """Data layer used to represent boxes (rectangular bounding boxes).

    Parameters
    ----------
    data: A Numpy array of shape (N, 4, D) containing the coordinates of the four corners of the box.
    """

    kind = "boxes"
    mergers: Dict[str, Type[Merger]] = {
        "default": ObjectTileMerger,
        "override": ObjectMerger,
    }

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Boxes",
        description="Bounding boxes.",
        dimensionality: Optional[List[int]] = None,
        merger: str = "default",
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
            dimensionality=dimensionality,
            merger=merger,
            **kwargs,
        )

    @property
    def data_global_coords(self) -> Optional[np.ndarray]:
        """Data in global coordinate reference instead of local to the tile."""
        if (self.data is not None) and (self.tile_meta is not None):
            if self.tile_meta.coords_min is not None:
                data_global_coords = self.data.copy()
                for dim in range(self.ndim):
                    data_global_coords[:, :, dim] = data_global_coords[:, :, dim] + self.tile_meta.coords_min[dim]
                return data_global_coords

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        if self.data is not None:
            if self.n_objects > 0:
                return tuple(np.max(np.asarray(self.data).tolist(), axis=(0, 1)))

    def get_tile(self, tile_meta: TileMeta) -> Boxes:
        if self.data is None:
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self._get_initial_data(self.pixel_domain)
            _meta = self.meta
        else:
            # Mask of box coordinates in the tile
            boxes_coords_in_tile = (self.data >= tile_meta.coords_min) & (
                self.data < tile_meta.coords_max
            )

            # All coordinates must be in the tile bounds
            tile_filter = boxes_coords_in_tile.reshape((len(boxes_coords_in_tile), -1)).all(
                axis=1
            )

            # Select boxes in the tile
            boxes_tile_data = self.data[tile_filter]

            # Select meta of boxes in the tile
            boxes_tile_meta = extract_meta_tile(self.meta, self.n_objects, tile_filter)
            
            if boxes_tile_data is not None:
                boxes_tile_data = boxes_tile_data - tile_meta.coords_min
                
            _data = boxes_tile_data
            _meta = boxes_tile_meta
        
        return Boxes(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=tile_meta,
        )

    @staticmethod
    def _get_initial_data(
        pixel_domain: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros((0, 4, len(pixel_domain)), dtype=np.float32)