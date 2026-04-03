from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core.tiling import Domain


class Image(Layer):
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
        rgb: bool = False,
        **kwargs,
    ):
        super().__init__(
            data=data,
            name=name,
            description=description,
            dimensionality=dimensionality,
            rgb=rgb,
            **kwargs,
        )

    @property
    def bounds(self) -> Optional[Tuple]:
        """Data bounds in local coordinates."""
        if self._data is None:
            return
        if self.meta is None:
            return
        if self.meta["rgb"] is True:
            return (self._data.shape[0], self._data.shape[1])
        else:
            return self._data.shape

    def select(self, domain: Domain) -> Image:
        """Select data in a given domain."""
        if (
            (self.data is None)
            or (domain.coords_max is None)
            or (self.coords_max is None)
        ):
            _data = None
        else:
            domain_local = domain.copy()
            domain_local.coords_min = tuple(
                np.array(domain_local.coords_min) - np.array(self.coords_min)
            )
            try:
                _data = self.data[domain_local.slices]
            except:
                raise RuntimeError(
                    "Data re-initialization in the provided domain failed. Did you pass a domain range outside of the object's domain?"
                )

        return Image(
            data=_data,
            name=self.name,
            meta=self.meta,
            tile_meta=self.tile_meta,
            domain=domain,
        )

    def zeros_in(self, domain: Optional[Domain]) -> Optional[np.ndarray]:
        """Initialize zero-valued data in a given domain."""
        if domain is not None:
            return np.zeros(domain.size, dtype=np.float32)

    def initialize(self, domain_size: List[int]) -> Optional[np.ndarray]:
        if self.meta["rgb"] is True:
            domain_size.extend([3])
        return np.zeros(domain_size, dtype=np.float32)

    def reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        domain_local = domain.copy()
        domain_local.coords_min = tuple(
            np.array(domain_local.coords_min) - np.array(self.coords_min)
        )
        try:
            self.data[domain_local.slices] = 0
        except:
            raise RuntimeError(
                "Data re-initialization in the provided domain failed. Did you pass a domain range outside of the object's domain?"
            )
