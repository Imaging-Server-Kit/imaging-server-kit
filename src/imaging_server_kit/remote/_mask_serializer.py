from typing import List, Optional, Union

from geojson import Feature
import numpy as np

from imaging_server_kit.remote.encoding import decode_contents, encode_contents
from imaging_server_kit.remote.data_serializer import Serializer
from imaging_server_kit.types._mask import Mask, mask2features


class MaskDataSerializer(Serializer):
    @staticmethod
    def serialize(
        mask: Optional[Mask], client_origin: str
    ) -> Optional[Union[List[Feature], str]]:
        if mask is None:
            return
        
        if mask.data is None:
            return
        
        if client_origin == "Python/Napari":
            features = encode_contents(mask.data.astype(np.uint16))
        elif client_origin == "Java/QuPath":
            features = mask2features(mask.data)
        else:
            raise ValueError(f"Unrecognized client origin: {client_origin}")
        
        return features

    @staticmethod
    def deserialize(
        serialized_mask: Optional[Union[List[Feature], str]], client_origin: str
    ) -> Optional[np.ndarray]:
        if serialized_mask is None:
            return None
        if isinstance(serialized_mask, str):
            if client_origin == "Python/Napari":
                mask = decode_contents(serialized_mask).astype(int)
            else:
                raise ValueError(f"Unrecognized client origin: {client_origin}")
        
        return mask
