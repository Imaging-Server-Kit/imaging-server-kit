from __future__ import annotations

from typing import List, Optional
import numpy as np

from imaging_server_kit.types._objects import _ObjectsLayer


class Points(_ObjectsLayer):
    """Data layer for sets of points (2D, 3D).

    Parameters
    ----------
    data : numpy.ndarray, optional
        Point coordinates, as an array of shape `(N, D)`, where `D` is the number of
        dimensions.
    name : str, default="Points"
        Name of the layer.
    description : str, default="Input points (2D, 3D)"
        Description of the layer, displayed on the algorithm documentation page.
    dimensionality : list of int, optional
        Accepted numbers of dimensions, for example `[2, 3]`. By default, any number
        of dimensions is accepted.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).
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
