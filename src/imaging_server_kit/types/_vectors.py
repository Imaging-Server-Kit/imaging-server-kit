from __future__ import annotations

from typing import List, Optional
import numpy as np

from imaging_server_kit.types._objects import _ObjectsLayer


class Vectors(_ObjectsLayer):
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
    _object_shape = (2,)

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

    def _anchor(self, data: np.ndarray) -> np.ndarray:
        # Vectors are selected based on their origin
        return data[:, 0, :]

    def _offset(self, data: np.ndarray, offset) -> np.ndarray:
        # Only the origins are translated (not the displacements)
        data = data.copy()
        data[:, 0, :] = data[:, 0, :] + offset
        return data

    def _bounds_coords(self, data: np.ndarray) -> np.ndarray:
        return data[:, 0, :]
