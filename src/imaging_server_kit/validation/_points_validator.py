
from typing import Optional

import numpy as np

from imaging_server_kit.validation.layer_validator import Validator
from imaging_server_kit.types._points import Points


class PointsValidator(Validator):
    @staticmethod
    def validate(points: Optional[Points]) -> None:
        data = points.data
        
        assert isinstance(
            data, np.ndarray
        ), f"Points data ({type(data)}) is not a Numpy array"
        assert len(data.shape) == 2, "Points data should have shape (N, D)"
        