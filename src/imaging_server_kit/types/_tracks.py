from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer


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
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            data=data,
            meta=meta,
            tile_meta=tile_meta,
            description=description,
            dimensionality=dimensionality,
            **kwargs,
        )

    @staticmethod
    def initialize_data(
        bounds: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if bounds is None:
            return
        return np.zeros((1, len(bounds) + 2), dtype=np.float32)

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def data_bounds(self) -> Optional[Tuple]:
        if self.data is None:
            if self.n_objects > 0:
                return tuple(np.max(self.data, axis=0)[2:])
