
from typing import Optional

import numpy as np

from imaging_server_kit.validation.layer_validator import Validator
from imaging_server_kit.types._mask import Mask


class MaskValidator(Validator):
    @staticmethod
    def validate(mask: Optional[Mask]) -> None:
        data = mask.data
        meta = mask.meta
        
        assert isinstance(
            data, np.ndarray
        ), f"Mask data ({type(data)}) is not a Numpy array"

        if not all(data.shape):
            raise ValueError("Image array has an invalid shape: ", data.shape)

        if len(data.shape) not in meta["dimensionality"]:
            raise ValueError("Image array has the wrong dimensionality.")
        