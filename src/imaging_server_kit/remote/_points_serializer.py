from typing import Any, Optional

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import decode_coords, encode_coords
import numpy as np

from imaging_server_kit.types._points import Points


class PointsDataSerializer(Serializer):
    @staticmethod
    def serialize(points: Optional[Points]) -> Any:
        if points is None:
            return

        if points.data is None:
            return

        return encode_coords(points.data)

    @staticmethod
    def deserialize(serialized_points: Any) -> Optional[np.ndarray]:
        return decode_coords(serialized_points)
