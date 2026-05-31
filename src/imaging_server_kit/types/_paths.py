from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core.tiling import Domain


class Paths(Layer):
    """Data layer used to represent 2D and 3D paths.

    Parameters
    ----------
    data: A list of Numpy arrays (one for each path), each with shape (N, D),
        where N is the length (number of points) in the path and D the dimensionality (2, 3..).
    """

    kind = "paths"

    def __init__(
        self,
        data: Optional[List] = None,
        name="Paths",
        description="Input paths shapes (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        serializer: str = "default",
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            data=data,
            dimensionality=dimensionality,
            serializer=serializer,
            **kwargs,
        )

    def __str__(self) -> str:
        return f"{self.name} ({self.kind} layer). Paths: {self.n_objects}"

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
            path_bounds = []
            for path in self.data:
                path_bounds.append(list(np.max(path, axis=0)))
            bounds_min = tuple(np.min(np.asarray(path_bounds), axis=0).tolist())
            bounds_max = tuple(np.max(np.asarray(path_bounds), axis=0).tolist())

            return (bounds_min, bounds_max)

    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            return np.asarray([])
