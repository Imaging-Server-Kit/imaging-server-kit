from typing import Any, List, Optional

import numpy as np

from imaging_server_kit.remote.encoding import decode_coords, encode_coords
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.types._paths import Paths


class PathsDataSerializer(Serializer):
    @staticmethod
    def serialize(paths: Optional[Paths]) -> Optional[List[Any]]:
        if paths is None:
            return

        if paths.data is not None:
            return [encode_coords(arr) for arr in paths.data]

    @staticmethod
    def deserialize(
        serialized_paths: Optional[List[Any]],
    ) -> Optional[List[np.ndarray]]:
        if serialized_paths is None:
            return

        data = []
        for f in serialized_paths:
            path = decode_coords(f)
            if path is not None:
                data.append(path)

        return data
