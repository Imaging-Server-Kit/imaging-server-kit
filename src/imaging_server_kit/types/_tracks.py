from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core.tiling import Domain


class Tracks(Layer):
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

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def _bounds(self) -> Optional[Tuple]:
        """Data bounds in local coordinates, given the data."""
        if (self.data is None) or (self.n_objects == 0):
            return

        # Coordinates are [T, (Z), Y, X] (the first column is the track ID)
        bounds_min = tuple(np.min(self.data, axis=0)[1:].tolist())
        bounds_max = tuple(np.max(self.data, axis=0)[1:].tolist())

        return (bounds_min, bounds_max)
