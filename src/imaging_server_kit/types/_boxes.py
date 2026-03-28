from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import Domain
from imaging_server_kit.types.common import extract_meta_tile
from imaging_server_kit.types.data_layer import DataLayer


class Boxes(DataLayer):
    """Data layer used to represent boxes (rectangular bounding boxes).

    Parameters
    ----------
    data: A Numpy array of shape (N, 4, D) containing the coordinates of the four corners of the box.
    """

    kind = "boxes"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Boxes",
        description="Bounding boxes.",
        dimensionality: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            name=name,
            description=description,
            dimensionality=dimensionality,
            **kwargs,
        )

    @property
    def data_global_coords(self) -> Optional[np.ndarray]:
        """Data in global coordinate reference instead of local to the tile."""
        if (self.data is not None) and (self.domain is not None):
            if self.domain.coords_min is not None:
                data_global_coords = self.data.copy()
                for dim in range(self.ndim):
                    data_global_coords[:, :, dim] = (
                        data_global_coords[:, :, dim] + self.domain.coords_min[dim]
                    )
                return data_global_coords

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def _data_bounds(self) -> Optional[Tuple]:
        if self.data is not None:
            if self.n_objects > 0:
                return tuple(np.max(np.asarray(self.data).tolist(), axis=(0, 1)))

    def select(self, domain: Domain) -> Boxes:
        if self.data is None:
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self.initialize_data(self.coords_max)
            _meta = self.meta
        else:
            # Mask of box coordinates in the tile
            boxes_coords_in_tile = (self.data_global_coords >= domain.coords_min) & (
                self.data_global_coords < domain.coords_max
            )

            # All coordinates must be in the tile bounds
            tile_filter = boxes_coords_in_tile.reshape(
                (len(boxes_coords_in_tile), -1)
            ).all(axis=1)

            # Select boxes in the tile
            boxes_tile_data = self.data[tile_filter]

            # Select meta of boxes in the tile
            boxes_tile_meta = extract_meta_tile(self.meta, self.n_objects, tile_filter)

            if len(boxes_tile_data) > 0:
                btd = boxes_tile_data.copy()
                for dim in range(self.ndim):
                    btd[:, :, dim] = btd[:, :, dim] + (self.domain.coords_min[dim] - domain.coords_min[dim])
                boxes_tile_data = btd

            _data = boxes_tile_data
            _meta = boxes_tile_meta

        return Boxes(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta,
            domain=domain,
        )

    @staticmethod
    def initialize_data(
        bounds: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if bounds is None:
            return
        return np.zeros((0, 4, len(bounds)), dtype=np.float32)
