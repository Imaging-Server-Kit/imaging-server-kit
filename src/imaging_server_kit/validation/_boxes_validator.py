
from typing import Optional

import numpy as np

from imaging_server_kit.validation.layer_validator import Validator
from imaging_server_kit.types._boxes import Boxes


class BoxesValidator(Validator):
    @staticmethod
    def validate(boxes: Optional[Boxes]) -> None:
        data = boxes.data
        meta = boxes.meta
        
        assert isinstance(
            data, np.ndarray
        ), f"Boxes data ({type(data)}) is not a Numpy array"

        assert len(data.shape) == 3, "Boxes data should have shape (N, 4, D)"

        allowed_dims = meta["dimensionality"]
        assert (
            data.shape[2] in allowed_dims
        ), f"Boxes have an unsupported dimensionality: {data.shape[2]} (accepted: {allowed_dims})"
        
        
        