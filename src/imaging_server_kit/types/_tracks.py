from typing import List, Optional
import numpy as np

from imaging_server_kit.core.domain import Domain
from imaging_server_kit.types._objects import _ObjectsLayer


class Tracks(_ObjectsLayer):
    """Data layer for object tracks: point detections linked over time by a track ID.

    Parameters
    ----------
    data : numpy.ndarray, optional
        An array of shape `(N, D+1)`, with columns `[ID, T, (Z), Y, X]`.
    name : str, default="Tracks"
        Name of the layer.
    description : str, default="Input tracks (2D, 3D)"
        Description of the layer, displayed on the algorithm documentation page.
    dimensionality : list of int, optional
        Accepted numbers of dimensions, for example `[2, 3]`. By default, any number
        of dimensions is accepted.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).
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

    def _anchor(self, data: np.ndarray) -> np.ndarray:
        # Track points are selected by their coordinates [T, (Z), Y, X] (the first column is the track ID)
        return data[:, 1:]

    def _offset(self, data: np.ndarray, offset) -> np.ndarray:
        # The track IDs are not translated
        data = data.copy()
        data[:, 1:] = data[:, 1:] + offset
        return data

    def _bounds_coords(self, data: np.ndarray) -> np.ndarray:
        return data[:, 1:]

    def _zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            # Empty tracks: an ID column followed by the coordinates [T, (Z), Y, X]
            return np.zeros((0, domain.ndim + 1), dtype=np.float32)

    def _summary(self) -> str:
        if self.data is None:
            return "empty"
        n_tracks = len(np.unique(self.data[:, 0])) if len(self.data) else 0
        summary = f"{n_tracks} tracks ({self.n_objects} points)"
        if self.ndim is not None:
            summary += f", {self.ndim}D"
        return summary
