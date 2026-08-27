
from typing import Optional

import numpy as np

from imaging_server_kit.validation.layer_validator import Validator
from imaging_server_kit.types._boxes import Boxes


class BoxesValidator(Validator):
    @staticmethod
    def validate(boxes: Optional[Boxes]) -> None:
        if boxes is None:
            return
        
        data = boxes.data
        meta = boxes.meta
        
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Boxes data ({type(data)}) is not a Numpy array")
        
        if (len(data.shape) != 3) or (data.shape[1] != 4):
            raise ValueError("Boxes data should have shape (N, 4, D)")

        if meta:
            allowed_dims = meta["dimensionality"]
            
            if not data.shape[2] in allowed_dims:
                raise ValueError(f"Boxes have an unsupported dimensionality: {data.shape[2]} (accepted: {allowed_dims})")
        