from typing import Optional

import numpy as np

from imaging_server_kit.types import Image
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import decode_contents, encode_contents


class ImageDataSerializer(Serializer):
    @staticmethod
    def serialize(image: Optional[Image]) -> Optional[str]:
        if image is not None:
            image_data = image.data
            if image_data is not None:
                # Images are sent in their own dtype (TIFF preserves it)
                return encode_contents(np.asarray(image_data))

    @staticmethod
    def deserialize(serialized_data: Optional[str]) -> Optional[np.ndarray]:
        if serialized_data is not None:
            if isinstance(serialized_data, str):
                return decode_contents(serialized_data)
