from typing import Optional

import numpy as np

from imaging_server_kit.remote.data_serializer import Serializer
from imaging_server_kit.remote.encoding import decode_contents, encode_contents
from imaging_server_kit.types._tracks import Tracks


class TracksDataSerializer(Serializer):
    @staticmethod
    def serialize(tracks: Optional[Tracks], client_origin: str) -> Optional[str]:
        if tracks is None:
            return
        
        if tracks.data is None:
            return
        
        return encode_contents(tracks.data.astype(np.float32))
    
    @staticmethod
    def deserialize(
        serialized_tracks: Optional[str], client_origin: str
    ) -> Optional[np.ndarray]:
        if serialized_tracks is None:
            return
        
        if isinstance(serialized_tracks, str):
            tracks_data = decode_contents(serialized_tracks)
        
        return tracks_data.astype(float)