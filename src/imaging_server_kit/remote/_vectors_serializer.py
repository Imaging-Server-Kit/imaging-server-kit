from typing import Any, Optional

import numpy as np

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import decode_coords, encode_coords
from imaging_server_kit.types._vectors import Vectors


class VectorsDataSerializer(Serializer):
    @staticmethod
    def serialize(vectors: Optional[Vectors]) -> Any:
        if vectors is None:
            return

        if vectors.data is None:
            return

        return encode_coords(vectors.data)

    @staticmethod
    def deserialize(serialized_vectors: Any) -> Optional[np.ndarray]:
        return decode_coords(serialized_vectors)
