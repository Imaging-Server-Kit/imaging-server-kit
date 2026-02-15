from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer, DataSerializer


class TracksDataSerializer(DataSerializer):
    def serialize(self, tracks: Optional[np.ndarray], client_origin: str) -> Optional[str]:
        if tracks is not None:
            return encode_contents(tracks.astype(np.float32))
    
    def deserialize(self, serialized_tracks: Optional[str], client_origin: str) -> Optional[np.ndarray]:
        if serialized_tracks is None:
            return None
        if isinstance(serialized_tracks, str):
            serialized_tracks = decode_contents(serialized_tracks)
        return serialized_tracks.astype(float)
    

class Tracks(DataLayer):
    """Data layer used to represent tracking data.

    Parameters
    ----------
    data: A Numpy array of shape (N, D+1) where the dimensions (D) are [ID, T, (Z), Y, X].
    """

    kind = "tracks"
    data_serializers: Dict[str, Type[DataSerializer]] = {"default": TracksDataSerializer}

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Tracks",
        description="Input tracks (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        required: bool = True,
        data_serializer: str = "default",
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
            required=required,
            data_serializer=data_serializer,
            **kwargs,
        )

    @classmethod
    def _get_initial_data(
        cls, pixel_domain: Optional[Union[Tuple, List]]
    ) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros((1, len(pixel_domain) + 2), dtype=np.float32)

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        if self.data is None:
            if self.n_objects > 0:
                return tuple(np.max(self.data, axis=0)[2:])
