from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.core.tiling import Domain
from imaging_server_kit.types.common import select_object_meta
from imaging_server_kit.types.layer import Layer


class Boxes(Layer):
    """Data layer used to represent boxes (rectangular bounding boxes).

    Parameters
    ----------
    data: A Numpy array of shape (N, 4, D) containing the coordinates of the four corners of the box.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
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
        """Data in global coordinates."""
        if self.data is not None:
            return self.data + self.position
    
    def data_from_coords(self, coords: Tuple) -> Optional[np.ndarray]:
        if self.data is not None:
            return self.data + (np.asarray(self.position) - np.asarray(coords))

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def bounds(self) -> Optional[Tuple]:
        """Data bounds in local coordinates."""
        if self.data is None:
            return

        if self.n_objects == 0:
            return
        
        bounds_min = tuple(np.min(np.asarray(self.data).tolist(), axis=(0, 1)))
        bounds_max = tuple(np.max(np.asarray(self.data).tolist(), axis=(0, 1)))

        return (bounds_min, bounds_max)

    def select(self, domain: Domain) -> Boxes:
        """Select data in a given domain."""
        if (self.data is None) or (domain.size is None):
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self.zeros_in(domain=domain)
            _meta = self.meta
        else:
            # Mask of box coordinates in the tile
            boxes_in_domain = (self.data_global_coords >= domain.coords_min) & (
                self.data_global_coords < domain.coords_max
            )

            # All coordinates must be in the tile bounds
            filt = boxes_in_domain.reshape((len(boxes_in_domain), -1)).all(axis=1)

            selected_boxes = self.data_global_coords[filt]

            selected_meta = select_object_meta(self.meta, self.n_objects, filt)

            if len(selected_boxes) > 0:
                selected_boxes = selected_boxes - domain.coords_min

            _data = selected_boxes
            _meta = selected_meta

        return Boxes(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta,
            position=domain.coords_min,
        )

    def zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            return np.zeros((0, 4, domain.ndim), dtype=np.float32)

    def reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        if self.data is None:
            return

        if self.n_objects == 0:
            return

        objects_in_domain = (self.data_global_coords >= domain.coords_min) & (
            self.data_global_coords <= domain.coords_max
        )

        filt = objects_in_domain.reshape((len(objects_in_domain), -1)).all(axis=1)

        if len(self.data[filt]) > 0:
            self.data = self.data[~filt]
            self.meta = select_object_meta(self.meta, len(~filt), ~filt)
