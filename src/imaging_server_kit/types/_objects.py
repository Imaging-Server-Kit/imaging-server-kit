from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from imaging_server_kit.core.domain import Domain
from imaging_server_kit.types.layer import Layer
from imaging_server_kit.types.common import copy_meta, objects_in_domain, select_object_meta


class _ObjectsLayer(Layer):
    """Base class of layers holding a set of objects with coordinates: Points, Boxes, Vectors and Tracks.

    The data is an array of shape `(N, *_object_shape, D)`. Subclasses define how objects are
    anchored (for selection), offset, and bounded.
    """

    # Shape of a single object, without the coordinates axis, e.g. (4,) for the corners of boxes
    _object_shape: Tuple[int, ...] = ()

    def _anchor(self, data: np.ndarray) -> np.ndarray:
        """(N, D) coordinates deciding in which domain each object is selected."""
        return data

    def _offset(self, data: np.ndarray, offset) -> np.ndarray:
        """Data translated by `offset` (a new array)."""
        return data + offset

    def _bounds_coords(self, data: np.ndarray) -> np.ndarray:
        """(M, D) coordinates covered by the objects, used for the bounds."""
        return data.reshape((-1, data.shape[-1]))

    @property
    def data_global_coords(self) -> Optional[np.ndarray]:
        """Data in global coordinates."""
        if self.data is not None:
            return self._offset(self.data, self.position)

    def data_from_coords(self, coords: Tuple) -> Optional[np.ndarray]:
        """Data viewed from the given origin coordinates."""
        if self.data is not None:
            return self._offset(
                self.data, np.asarray(self.position) - np.asarray(coords)
            )

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
        coords = self._bounds_coords(self.data)
        bounds_min = tuple(np.min(coords, axis=0).tolist())
        bounds_max = tuple((np.max(coords, axis=0) + 1).tolist())

        return (bounds_min, bounds_max)

    def select(self, domain: Domain) -> _ObjectsLayer:
        """Select the part of the layer inside a domain.

        Parameters
        ----------
        domain : Domain
            The region to select, in global pixel coordinates.

        Returns
        -------
        Layer
            A new layer with the selected data, positioned in global coordinates.
        """
        cls = type(self)

        if domain.size is None:
            # Undefined domain: nothing to select, the data is kept as-is
            return cls(
                data=self.data,
                name=self.name,
                meta=copy_meta(self),
                tile_meta=self.tile_meta.copy(),
            )

        if self.n_objects == 0:
            _data = self._zeros_in(domain=domain)
            _meta = copy_meta(self)
        else:
            # Select objects via global coordinates
            data_global = self.data_global_coords
            filt = objects_in_domain(self._anchor(data_global), domain)  # (N,)

            selected = data_global[filt]

            if self.meta:
                selected_meta = select_object_meta(self.meta.copy(), self.n_objects, filt)
            else:
                selected_meta = self.meta

            if len(selected) > 0:
                # Return the objects in coordinates local to the selection domain
                selected = self._offset(selected, -np.asarray(domain.coords_min))

            _data = selected
            _meta = selected_meta

        selection = cls(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta.copy(),
        )

        selection.position = domain.coords_min

        return selection

    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            return np.zeros((0, *self._object_shape, domain.ndim), dtype=np.float32)

    def _reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        if self.data is None:
            return

        if self.n_objects == 0:
            return

        filt = objects_in_domain(self._anchor(self.data_global_coords), domain)  # (N,)

        if filt.any():
            self.data = self.data[~filt]
            self.meta = select_object_meta(self.meta, len(filt), ~filt)
