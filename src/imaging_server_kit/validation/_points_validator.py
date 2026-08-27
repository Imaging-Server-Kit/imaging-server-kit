
from typing import Optional

import numpy as np

from imaging_server_kit.validation.layer_validator import Validator
from imaging_server_kit.types._points import Points


class PointsValidator(Validator):
    @staticmethod
    def validate(points: Optional[Points]) -> None:
        if points is None:
            return
        
        data = points.data
        
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Points data ({type(data)}) is not a Numpy array")
        
        if len(data.shape) != 2:
            raise ValueError("Points data should have shape (N, D)")
        