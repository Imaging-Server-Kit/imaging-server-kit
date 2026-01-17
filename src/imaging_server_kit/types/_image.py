from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.data_layer import DataLayer
from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types._null import Null


class Image(DataLayer):
    """Data layer used to represent images and image-like data.

    Parameters
    ----------
    data: Numpy arrays.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    rgb: Set to True for RGB images.
    """

    kind = "image"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Image",
        description="Input image (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        required: bool = True,  # When set to True, triggers a parameter validation error if image is None
        rgb: bool = False,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
    ):
        self.rgb = rgb
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
        )
        self.dimensionality = (
            dimensionality if dimensionality is not None else np.arange(6).tolist()
        )
        self.required = required

        # Schema contributions
        main = {}
        if not self.required:
            self.default = None
            main["default"] = self.default
        extra = {
            "dimensionality": self.dimensionality,
            "rgb": self.rgb,
            "required": self.required,
        }
        self.constraints = [main, extra]

        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

    @property
    def _pixel_domain(self) -> Optional[Tuple]:
        if self.data is None:
            return
        if self.rgb:
            return (self.data.shape[0], self.data.shape[1])
        else:
            return self.data.shape

    def get_tile(self, tile_meta: TileMeta) -> Image:
        if (
            (self.data is None)
            or (tile_meta.coords_max is None)
            or (self.pixel_domain is None)
        ):
            _data = None
        elif (tile_meta.coords_max > np.asarray(self.pixel_domain)).any():
            _data = None
        else:
            _data = self.data[tile_meta.slices]
        return Image(
            data=_data,
            name=self.name,
            meta=self.meta,
            tile_meta=tile_meta,
        )

    def merge(self, image_tile: Image) -> None:
        if (self.data is None) and (image_tile.data is not None):
            self.data = image_tile.data
        elif (image_tile.data is None) or (image_tile.tile_meta is None):
            return
        else:
            _slices = image_tile.tile_meta.slices
            _overlap_count_map = image_tile.tile_meta.overlap_count_map
            if (_slices is not None) and (_overlap_count_map is not None):
                self.data[_slices] = (
                    self.data[_slices] + image_tile.data / _overlap_count_map
                )

    @classmethod
    def serialize(cls, data: Optional[np.ndarray], client_origin: str):
        if data is not None:
            return encode_contents(data.astype(np.float32))

    @classmethod
    def deserialize(
        cls, serialized_data: Optional[Union[np.ndarray, str]], client_origin: str
    ):
        if serialized_data is None:
            return None
        if isinstance(serialized_data, str):
            serialized_data = decode_contents(serialized_data)
        return serialized_data.astype(float)

    @classmethod
    def _get_initial_data(
        cls, pixel_domain: Optional[Union[Tuple, List]]
    ) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros(pixel_domain, dtype=np.float32)

    @classmethod
    def initialize(cls, pixel_domain: Union[Tuple, List]) -> Image:
        return cls(data=cls._get_initial_data(pixel_domain))

    @classmethod
    def validate_data(cls, data, meta, constraints):
        main, extra = constraints
        if extra["required"] is False:
            return

        assert isinstance(
            data, np.ndarray
        ), f"Image data ({type(data)}) is not a Numpy array"

        if not all(data.shape):
            raise ValueError("Image array has an invalid shape: ", data.shape)

        if len(data.shape) not in extra["dimensionality"]:
            raise ValueError("Image array has the wrong dimensionality.")

        if extra["rgb"] is True:
            if len(data.shape) != 3:
                raise ValueError("Image should be RGB(A).")
            if data.shape[2] not in [3, 4]:
                raise ValueError("Image should be RGB(A).")
