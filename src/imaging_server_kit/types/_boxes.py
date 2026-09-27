from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.core.tiling import Domain
from imaging_server_kit.types.common import copy_meta, objects_in_domain, select_object_meta
from imaging_server_kit.types.layer import Layer


class Boxes(Layer):
    """Data layer used to represent rectangular bounding boxes (2D, 3D).

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
        description="Bounding boxes",
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
    def _bounds(self) -> Optional[Tuple]:
        """Data bounds in local coordinates."""
        if self.data is None:
            return

        if self.n_objects == 0:
            return

        # The upper bound is exclusive (+1), like an image extent is its shape
        bounds_min = tuple(np.min(self.data, axis=(0, 1)).tolist())
        bounds_max = tuple((np.max(self.data, axis=(0, 1)) + 1).tolist())

        return (bounds_min, bounds_max)

    def select(self, domain: Domain) -> Boxes:
        """Select data in a given domain."""
        if domain.size is None:
            # Undefined domain: nothing to select, the data is kept as-is
            return Boxes(
                data=self.data,
                name=self.name,
                meta=copy_meta(self),
                tile_meta=self.tile_meta.copy(),
            )

        if self.n_objects == 0:
            _data = self._zeros_in(domain=domain)
            _meta = copy_meta(self)
        else:
            # Boxes are selected based on their center (each box goes to a single tile);
            # the selected boxes keep their full coordinates, which may extend beyond the domain.
            filt = objects_in_domain(self.data_global_coords.mean(axis=1), domain)  # (N,)

            selected_boxes = self.data_global_coords[filt]

            if self.meta:
                selected_meta = select_object_meta(self.meta.copy(), self.n_objects, filt)
            else:
                selected_meta = self.meta

            if len(selected_boxes) > 0:
                selected_boxes = selected_boxes - domain.coords_min

            _data = selected_boxes
            _meta = selected_meta

        boxes_selection = Boxes(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta.copy(),
        )
        
        boxes_selection.position = domain.coords_min
        
        return boxes_selection

    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            return np.zeros((0, 4, domain.ndim), dtype=np.float32)

    def _reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        if self.data is None:
            return

        if self.n_objects == 0:
            return

        filt = objects_in_domain(self.data_global_coords.mean(axis=1), domain)  # (N,)

        if filt.any():
            self.data = self.data[~filt]
            self.meta = select_object_meta(self.meta, len(filt), ~filt)
