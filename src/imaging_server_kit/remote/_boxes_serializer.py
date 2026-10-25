from typing import Any, Optional

import numpy as np

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import decode_coords, encode_coords
from imaging_server_kit.types._boxes import Boxes


class BoxesDataSerializer(Serializer):
    @staticmethod
    def serialize(boxes: Optional[Boxes]) -> Any:
        if boxes is None:
            return

        if boxes.data is None:
            return

        return encode_coords(boxes.data)

    @staticmethod
    def deserialize(serialized_boxes: Any) -> Optional[np.ndarray]:
        return decode_coords(serialized_boxes)
