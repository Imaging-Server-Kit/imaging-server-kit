
from typing import Optional

import numpy as np

from imaging_server_kit.validation.validator import Validator
from imaging_server_kit.types._points import Points


class PointsValidator(Validator):
    @staticmethod
    def validate(points: Optional[Points]) -> None:
        if points is None:
            return
        
        data = points.data
        meta = points.meta
        
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Points data ({type(data)}) is not a Numpy array")
        
        if len(data.shape) != 2:
            raise ValueError("Points data should have shape (N, D)")

        if meta:
            allowed_dims = meta["dimensionality"]

            if data.shape[1] not in allowed_dims:
                raise ValueError(f"Points have an unsupported dimensionality: {data.shape[1]} (accepted: {allowed_dims})")
        