from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.types.data_layer import DataLayer
from imaging_server_kit.core.tiling import Domain


class Tracks(DataLayer):
    """Data layer used to represent tracking data.

    Parameters
    ----------
    data: A Numpy array of shape (N, D+1) where the dimensions (D) are [ID, T, (Z), Y, X].
    """

    kind = "tracks"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Tracks",
        description="Input tracks (2D, 3D)",
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

    def zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            return np.zeros((1, self.ndim + 2), dtype=np.float32)

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def bounds(self) -> Optional[Tuple]:
        """Data bounds in local coordinates."""
        if self.data is None:
            if self.n_objects > 0:
                return tuple(np.max(self.data, axis=0)[2:])
