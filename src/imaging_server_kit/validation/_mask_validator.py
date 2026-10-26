
from typing import Optional

import numpy as np

from imaging_server_kit.validation.validator import Validator
from imaging_server_kit.types._mask import Mask


class MaskValidator(Validator):
    @staticmethod
    def validate(mask: Optional[Mask]) -> None:
        if mask is None:
            return
        
        data = mask.data
        meta = mask.meta
        
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Mask data ({type(data)}) is not a Numpy array")

        if not all(data.shape):
            raise ValueError("Mask array has an invalid shape: ", data.shape)

        if meta:
            if len(data.shape) not in meta["dimensionality"]:
                raise ValueError("Mask array has the wrong dimensionality.")
        