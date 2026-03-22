from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer


class Paths(DataLayer):
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
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
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
    def data_bounds(self) -> Optional[Tuple]:
        if self.data is None:
            return
        if self.n_objects > 0:
            path_bounds = []
            for path in self.data:
                path_bounds.append(list(np.max(path, axis=0)))
            bounds = np.max(np.asarray(path_bounds), axis=0).tolist()
            return tuple(bounds)

    @staticmethod
    def initialize_data(
        bounds: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if bounds is None:
            return
        return np.asarray([])
