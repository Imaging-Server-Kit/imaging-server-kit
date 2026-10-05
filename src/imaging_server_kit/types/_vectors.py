from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.core.tiling import Domain
from imaging_server_kit.types.layer import Layer
from imaging_server_kit.types.common import copy_meta, objects_in_domain, select_object_meta


class Vectors(Layer):
    """Data layer for sets of vectors (2D, 3D).

    Each vector is defined by an origin point and a displacement.

    Parameters
    ----------
    data : numpy.ndarray, optional
        An array of shape `(N, 2, D)`, where `D` is the number of dimensions.
        `data[:, 0, :]` holds the origins of the vectors, and `data[:, 1, :]` the
        displacements from the origins.
    name : str, default="Vectors"
        Name of the layer.
    description : str, default="Input vectors (2D, 3D)"
        Description of the layer, displayed on the algorithm documentation page.
    dimensionality : list of int, optional
        Accepted numbers of dimensions, for example `[2, 3]`. By default, any number
        of dimensions is accepted.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).
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

        # The upper bound is exclusive (+1), like an image extent is its shape
        bounds_min = tuple(np.min(self.data[:, 0, :], axis=0).tolist())
        bounds_max = tuple((np.max(self.data[:, 0, :], axis=0) + 1).tolist())

        return (bounds_min, bounds_max)

    def select(self, domain: Domain) -> Vectors:
        """Select the part of the layer inside a domain.

        Parameters
        ----------
        domain : Domain
            The region to select, in global pixel coordinates.

        Returns
        -------
        Vectors
            A new layer with the selected data, positioned in global coordinates.
        """
        if domain.size is None:
            # Undefined domain: nothing to select, the data is kept as-is
            return Vectors(
                data=self.data,
                name=self.name,
                meta=copy_meta(self),
                tile_meta=self.tile_meta.copy(),
            )

        if self.n_objects == 0:
            _data = self._zeros_in(domain=domain)
            _meta = copy_meta(self)
        else:
            # Vectors are selected based on their origin
            filt = objects_in_domain(self.data_global_coords[:, 0, :], domain)  # (N,)

            selected_vectors = self.data_global_coords[filt]

            if self.meta:
                selected_meta = select_object_meta(self.meta.copy(), self.n_objects, filt)
            else:
                selected_meta = self.meta

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
            tile_meta=self.tile_meta.copy(),
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

        filt = objects_in_domain(self.data_global_coords[:, 0, :], domain)  # (N,)

        if filt.any():
            self.data = self.data[~filt]
            self.meta = select_object_meta(self.meta, len(filt), ~filt)
