from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.core.domain import Domain
from imaging_server_kit.types.common import copy_meta, objects_in_domain, select_object_meta
from imaging_server_kit.types.layer import Layer


class Paths(Layer):
    """Data layer for paths (2D, 3D), for example spline curves.

    Parameters
    ----------
    data : list of numpy.ndarray, optional
        One array per path, each of shape `(N, D)`, where `N` is the number of points
        in the path and `D` the number of dimensions.
    name : str, default="Paths"
        Name of the layer.
    description : str, default="Input paths (2D, 3D)"
        Description of the layer, displayed on the algorithm documentation page.
    dimensionality : list of int, optional
        Accepted numbers of dimensions, for example `[2, 3]`. By default, any number
        of dimensions is accepted.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).
    """

    kind = "paths"

    def __init__(
        self,
        data: Optional[List] = None,
        name="Paths",
        description="Input paths (2D, 3D)",
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
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def _bounds(self) -> Optional[Tuple]:
        """Data bounds in local coordinates, given the data."""
        if self.data is None:
            return
        if self.n_objects > 0:
            paths_min = [np.min(path, axis=0) for path in self.data]
            paths_max = [np.max(path, axis=0) for path in self.data]
            # The upper bound is exclusive (+1), like an image extent is its shape
            bounds_min = tuple(np.min(np.asarray(paths_min), axis=0).tolist())
            bounds_max = tuple((np.max(np.asarray(paths_max), axis=0) + 1).tolist())

            return (bounds_min, bounds_max)

    def select(self, domain: Domain) -> Paths:
        """Select the part of the layer inside a domain.

        Paths are selected based on their center (each path goes to a single tile); the selected
        paths keep all their points, which may extend beyond the domain.

        Parameters
        ----------
        domain : Domain
            The region to select, in global pixel coordinates.

        Returns
        -------
        Paths
            A new layer with the selected paths, positioned in global coordinates.
        """
        if domain.size is None:
            # Undefined domain: nothing to select, the data is kept as-is
            return Paths(
                data=self.data,
                name=self.name,
                meta=copy_meta(self),
                tile_meta=self.tile_meta.copy(),
            )

        if self.n_objects == 0:
            _data = self.data
            _meta = copy_meta(self)
        else:
            # Select paths via the global coordinates of their centers
            offset = np.asarray(self.position)
            centers = np.array([np.mean(path, axis=0) for path in self.data]) + offset
            filt = objects_in_domain(centers, domain)  # (N,)

            # Return the paths in coordinates local to the selection domain
            local_offset = offset - np.asarray(domain.coords_min)
            _data = [
                np.asarray(path) + local_offset
                for path, selected in zip(self.data, filt)
                if selected
            ]

            if self.meta:
                _meta = select_object_meta(self.meta.copy(), self.n_objects, filt)
            else:
                _meta = self.meta

        paths_selection = Paths(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=self.tile_meta.copy(),
        )

        paths_selection.position = domain.coords_min

        return paths_selection

    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            return np.asarray([])
