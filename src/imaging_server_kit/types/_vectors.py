from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.core.tiling import Domain
from imaging_server_kit.types.layer import Layer
from imaging_server_kit.types.common import select_object_meta


class Vectors(Layer):
    """Data layer used to represent vectors.

    Parameters
    ----------
    data: A Numpy array of shape (N, 2, D) where D is the dimensionality (2, 3..).
        data[:, 0, :] represents the coordinates of the origin of the vectors.
        data[:, 1, :] represents the displacement from the origin.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    """

    kind = "vectors"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Vectors",
        description="Input vectors (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            data=data,
            description=description,
            dimensionality=dimensionality,
            **kwargs,
        )

    @property
    def data_global_coords(self) -> Optional[np.ndarray]:
        """Data in global coordinates."""
        if self.data is not None:
            data_global = self.data.copy()
            data_global[:, 0, :] = data_global[:, 0, :] + self.position
            return data_global

    def data_from_coords(self, coords: Tuple) -> Optional[np.ndarray]:
        if self.data is not None:
            _data = self.data.copy()
            _data[:, 0, :] = _data[:, 0, :] + (
                np.asarray(self.position) - np.asarray(coords)
            )
            return _data

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

        bounds_min = tuple(np.min(self.data[:, 0, :], axis=0))
        bounds_max = tuple(np.max(self.data[:, 0, :], axis=0))

        return (bounds_min, bounds_max)

    def select(self, domain: Domain) -> Vectors:
        """Select data in a given domain."""
        if (self.data is None) or (domain.size is None):
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self._zeros_in(domain=domain)
            _meta = self.meta
        else:
            # Mask of vector coordinates in the domain
            vector_coords_in_domain = (
                self.data_global_coords[:, 0, :] >= domain.coords_min
            ) & (self.data_global_coords[:, 0, :] < domain.coords_max)

            # All coordinates must be in the domain bounds
            filt = vector_coords_in_domain.all(axis=1)  # (N,)

            selected_vectors = self.data_global_coords[filt]

            selected_meta = select_object_meta(self.meta, self.n_objects, filt)

            if len(selected_vectors) > 0:
                vtd = selected_vectors.copy()
                vtd[:, 0, :] = vtd[:, 0, :] - domain.coords_min
                selected_vectors = vtd

            _data = selected_vectors
            _meta = selected_meta

        vectors_selection = Vectors(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta,
        )
        
        vectors_selection.position = domain.coords_min
        
        return vectors_selection
        
    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            return np.zeros((0, 2, domain.ndim), dtype=np.float32)

    def _reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        if self.data is None:
            return

        if self.n_objects == 0:
            return

        objects_in_domain = (self.data_global_coords[:, 0] >= domain.coords_min) & (
            self.data_global_coords[:, 0] <= domain.coords_max
        )

        filt = objects_in_domain.all(axis=1)  # (N,)

        if len(self.data[filt]) > 0:
            self.data = self.data[~filt]
            self.meta = select_object_meta(self.meta, len(~filt), ~filt)
