from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.types.data_layer import DataLayer, DefaultMerger, DataSerializer
from imaging_server_kit.core.tiling import TileMeta


class ImageMerger(DefaultMerger):
    
    def merge(self, src_layer: Image, dst_layer: Image) -> None:
        if (
            (dst_layer.data is None)
            or (dst_layer.tile_meta is None)
            or (dst_layer.pixel_domain is None)
        ):
            return

        if (
            (src_layer.data is None)
            or (src_layer.tile_meta is None)
            or (src_layer.pixel_domain is None)
        ):
            # Will set src_layer.tile_meta and src_layer.pixel_domain
            src_layer.data = src_layer._get_initial_data(tuple([1]*dst_layer.ndim))
        
        if (src_layer.pixel_domain is None) or (src_layer.tile_meta is None):
            return  # This should never happen (just there for type hints)

        _s1 = dst_layer.tile_meta.slices
        _o1 = dst_layer.tile_meta.overlap_count_map
        if (_s1 is not None) and (_o1 is not None):
            _stack = np.stack([src_layer.pixel_domain, dst_layer.pixel_domain])
            _pixel_domain = np.max(_stack, axis=0).tolist()

            # If the incoming tile extends the pixel domain, we create a new Image, write src_layer.data into it, then merge the tile
            if _pixel_domain != src_layer.pixel_domain:
                new = Image.initialize(_pixel_domain)
                new_data = new.data
                new_data[src_layer.tile_meta.slices] = src_layer.data
            else:
                new_data = src_layer.data
            
            new_data[_s1] = new_data[_s1] + dst_layer.data / _o1  # type: ignore
            src_layer.data = new_data

    def first_tile_hook(self, src_layer: Image, dst_layer: Image):
        # Erase all of the data before tiling
        src_layer.data = src_layer._get_initial_data(src_layer.data_pixel_domain)


class ImageDataSerializer(DataSerializer):
    def serialize(self, data: Optional[np.ndarray], client_origin: str) -> Optional[str]:
        if data is not None:
            return encode_contents(data.astype(np.float32))
    
    def deserialize(self, serialized_data: Optional[str], client_origin: str) -> Optional[np.ndarray]:
        if serialized_data is not None:
            if isinstance(serialized_data, str):
                serialized_data = decode_contents(serialized_data)
                return serialized_data.astype(float)
            

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
        
        # Merging strategy
        self.merger = ImageMerger()
        
        # Serializer
        self.data_serializer = ImageDataSerializer()

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        if self._data is None:
            return
        if self.rgb:
            return (self._data.shape[0], self._data.shape[1])
        else:
            return self._data.shape

    def get_tile(self, tile_meta: TileMeta) -> Image:
        if (
            (self.data is None)
            or (tile_meta.coords_max is None)
            or (self.pixel_domain is None)
        ):
            _data = None
        else:
            _data = self.data[tile_meta.slices]
        return Image(
            data=_data,
            name=self.name,
            meta=self.meta,
            tile_meta=tile_meta,
        )

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
