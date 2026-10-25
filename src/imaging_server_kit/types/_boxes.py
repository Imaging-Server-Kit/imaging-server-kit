from __future__ import annotations

from typing import List, Optional
import numpy as np

from imaging_server_kit.types._objects import _ObjectsLayer


class Boxes(_ObjectsLayer):
    """Data layer for bounding boxes (2D, 3D).

    Boxes can be oriented (OBB).

    Parameters
    ----------
    data : numpy.ndarray, optional
        The coordinates of the four corners of each box, as an array of shape
        `(N, 4, D)`, where `D` is the number of dimensions.
    name : str, default="Boxes"
        Name of the layer.
    description : str, default="Bounding boxes"
        Description of the layer, displayed on the algorithm documentation page.
    dimensionality : list of int, optional
        Accepted numbers of dimensions, for example `[2, 3]`. By default, any number
        of dimensions is accepted.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).
    """

    kind = "boxes"
    _object_shape = (4,)

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

    def _anchor(self, data: np.ndarray) -> np.ndarray:
        # Boxes are selected based on their center (each box goes to a single tile);
        # the selected boxes keep their full coordinates, which may extend beyond the domain.
        return data.mean(axis=1)
