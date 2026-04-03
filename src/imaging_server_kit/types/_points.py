from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.core.tiling import Domain
from imaging_server_kit.types.common import select_object_meta
from imaging_server_kit.types.data_layer import DataLayer


class Points(DataLayer):
    """Data layer used to represent points.

    Parameters
    ----------
    data: A Numpy array of shape (N, D) representing point coordinates, where D is the dimensionality (2, 3..).
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    """

    kind = "points"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Points",
        description="Input points (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            data=data,
            dimensionality=dimensionality,
            **kwargs,
        )

    @property
    def data_global_coords(self) -> Optional[np.ndarray]:
        """Data in global coordinates."""
        if (self.data is not None) and (self.domain is not None):
            if self.domain.coords_min is not None:
                data_global = self.data.copy()
                for dim in range(self.ndim):
                    data_global[:, dim] = (
                        data_global[:, dim] + self.domain.coords_min[dim]
                    )
                return data_global

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def bounds(self) -> Optional[Tuple]:
        """Data bounds in local coordinates."""
        if self.data is not None:
            if self.n_objects > 0:
                return tuple(np.max(self.data, axis=0).tolist())

    def select(self, domain: Domain) -> Points:
        """Select data in a given domain."""
        if self.data is None:
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self.zeros_in(domain=self.domain)
            _meta = self.meta
        else:
            # Select points via global coordinates
            points_in_domain = (self.data_global_coords >= domain.coords_min) & (
                self.data_global_coords < domain.coords_max
            )

            # All coordinates must be in the tile bounds
            filt = points_in_domain.all(axis=1)  # (N,)

            selected_points = self.data[filt]

            selected_meta = select_object_meta(self.meta, self.n_objects, filt)

            if len(selected_points) > 0:
                # Adjust the data so that the origin is zero at the origin of `domain`
                ptd = selected_points.copy()
                for dim in range(self.ndim):
                    ptd[:, dim] = ptd[:, dim] + (
                        self.domain.coords_min[dim] - domain.coords_min[dim]
                    )
                selected_points = ptd

            _data = selected_points
            _meta = selected_meta

        return Points(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta,
            domain=domain,
        )

    def zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            return np.zeros((0, domain.ndim), dtype=np.float32)

    def reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        if self.data is None:
            return

        if self.n_objects == 0:
            return

        objects_in_domain = (self.data_global_coords >= domain.coords_min) & (
            self.data_global_coords < domain.coords_max
        )

        filt = objects_in_domain.all(axis=1)  # (N,)

        if len(self.data[filt]) > 0:
            self.data = self.data[~filt]
            self.meta = select_object_meta(self.meta, len(~filt), ~filt)
