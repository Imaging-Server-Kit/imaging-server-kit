from typing import Optional

import numpy as np

from imaging_server_kit.types import Image
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import decode_contents, encode_contents


class ImageDataSerializer(Serializer):
    @staticmethod
    def serialize(image: Optional[Image], client_origin: str) -> Optional[str]:
        if image is not None:
            image_data = image.data
            if image_data is not None:
                return encode_contents(image_data.astype(np.float32))

    @staticmethod
    def deserialize(serialized_data: Optional[str], client_origin: str) -> Optional[np.ndarray]:
        if serialized_data is not None:
            if isinstance(serialized_data, str):
                data = decode_contents(serialized_data)
                return data.astype(float)
