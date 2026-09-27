from typing import List, Optional, Union

from geojson import Feature
import numpy as np

from imaging_server_kit.remote.encoding import decode_contents, encode_contents
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.types._mask import Mask


class MaskDataSerializer(Serializer):
    @staticmethod
    def serialize(mask: Optional[Mask]) -> Optional[Union[List[Feature], str]]:
        if mask is None:
            return

        if mask.data is None:
            return

        # Masks are sent in their own dtype (TIFF preserves it)
        return encode_contents(np.asarray(mask.data))

    @staticmethod
    def deserialize(serialized_mask: Optional[str]) -> Optional[np.ndarray]:
        if isinstance(serialized_mask, str):
            return decode_contents(serialized_mask)
