
from typing import Optional

import numpy as np

from imaging_server_kit.validation.layer_validator import Validator
from imaging_server_kit.types._vectors import Vectors


class VectorsValidator(Validator):
    @staticmethod
    def validate(vectors: Optional[Vectors]) -> None:
        data = vectors.data
        assert isinstance(
            data, np.ndarray
        ), f"Vectors data ({type(data)}) is not a Numpy array"
        assert len(data.shape) == 3, "Vectors data should have shape (N, 2, D)"
        assert data.shape[1] == 2, "Vectors data should have shape (N, 2, D)"