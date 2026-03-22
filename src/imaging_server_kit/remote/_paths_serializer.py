from typing import List, Optional

import numpy as np

from imaging_server_kit.remote.encoding import decode_contents, encode_contents
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.types._paths import Paths


class PathsDataSerializer(Serializer):
    @staticmethod
    def serialize(paths: Optional[Paths], client_origin: str) -> Optional[List[str]]:
        if paths is None:
            return

        if paths.data is not None:
            return [encode_contents(arr.astype(np.float32)) for arr in paths.data]

    @staticmethod
    def deserialize(
        serialized_paths: Optional[List[str]], client_origin: str
    ) -> Optional[List[np.ndarray]]:
        if serialized_paths is None:
            return

        data = []
        for f in serialized_paths:
            if isinstance(f, str):
                f = decode_contents(f)
                data.append(f.astype(float))

        return data
