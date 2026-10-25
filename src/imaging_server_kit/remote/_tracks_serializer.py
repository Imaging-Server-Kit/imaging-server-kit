from typing import Any, Optional

import numpy as np

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import decode_coords, encode_coords
from imaging_server_kit.types._tracks import Tracks


class TracksDataSerializer(Serializer):
    @staticmethod
    def serialize(tracks: Optional[Tracks]) -> Any:
        if tracks is None:
            return

        if tracks.data is None:
            return

        return encode_coords(tracks.data)

    @staticmethod
    def deserialize(serialized_tracks: Any) -> Optional[np.ndarray]:
        return decode_coords(serialized_tracks)
