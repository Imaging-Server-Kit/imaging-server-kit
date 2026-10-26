from typing import Any, Optional, Union

import numpy as np

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import (
    decode_contents,
    decode_coords,
    encode_contents,
    encode_coords,
)
from imaging_server_kit.types import Layer


class ArrayDataSerializer(Serializer):
    """Data of array layers (Image, Mask), sent as TIFF in their own dtype."""

    @staticmethod
    def serialize(layer: Optional[Layer]) -> Optional[str]:
        if (layer is None) or (layer.data is None):
            return
        return encode_contents(np.asarray(layer.data))

    @staticmethod
    def deserialize(serialized_data: Optional[str]) -> Optional[np.ndarray]:
        if isinstance(serialized_data, str):
            return decode_contents(serialized_data)


class CoordsDataSerializer(Serializer):
    """Data of object layers (Points, Boxes, Vectors, Tracks), sent as float32 coordinates."""

    @staticmethod
    def serialize(layer: Optional[Layer]) -> Optional[Union[str, dict]]:
        if (layer is None) or (layer.data is None):
            return
        return encode_coords(layer.data)

    @staticmethod
    def deserialize(serialized_data: Any) -> Optional[np.ndarray]:
        return decode_coords(serialized_data)
