from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core.tiling import Domain
from imaging_server_kit.types.common import safe_index_slice


class Image(Layer):
    """Data layer used to represent images and image-like data.

    Parameters
    ----------
    data: Numpy arrays.
    dimensionality: list of accepted dimensionalities, for example [2, 3].
    rgb: Set to True for RGB images.
    channel_axis: Optional index of the channel axis.
      - The channel axis does not affect the `bounds`, `ndim`, and `domain` attributes.
      - The channel axis is set to `2` if rgb is True and there is no time axis.
      - tile_size along the channel axis defaults to the length of this axis.
    """

    kind = "image"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Image",
        description="Input image (2D, 3D)",
        dimensionality: Optional[List[int]] = None,
        rgb: bool = False,
        channel_axis: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            name=name,
            description=description,
            dimensionality=dimensionality,
            rgb=rgb,
            channel_axis=channel_axis,
            **kwargs,
        )

    @property
    def channel_axis(self):
        if self.meta["rgb"] is True:
            if self.data is not None:
                if self.data.ndim == 4:
                    return 3
                else:
                    return 2
            else:
                return 2

        if self.meta["channel_axis"] is not None:
            return self.meta["channel_axis"]

    @property
    def bounds(self) -> Optional[Tuple]:
        """Data bounds in local coordinates."""
        if self._data is None:
            return

        if self.meta is None:
            return

        if self.channel_axis is not None:
            shape = list(self._data.shape)
            shape.pop(self.channel_axis)
            return tuple(shape)
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
        elif (domain.coords_max > np.asarray(self.coords_max)).any():
            _data = None
        else:
            domain_local = domain.copy()
            domain_local.coords_min = tuple(
                np.array(domain_local.coords_min) - np.array(self.coords_min)
            )
            slices_int = tuple(safe_index_slice(s) for s in domain_local.slices)

            # Account for the channel_axis
            if self.channel_axis:
                slices_int_with_channel = (
                    slices_int[: self.channel_axis]
                    + (slice(None),)
                    + slices_int[self.channel_axis :]
                )
            else:
                slices_int_with_channel = slices_int

            try:
                _data = self.data[slices_int_with_channel]
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

    def reinitialize(self, domain: Domain) -> None:
        """Remove data in a given domain."""
        if not isinstance(domain, Domain):
            return

        domain_local = domain.copy()
        domain_local.coords_min = tuple(
            np.array(domain_local.coords_min) - np.array(self.coords_min)
        )
        slices_int = tuple(safe_index_slice(s) for s in domain_local.slices)

        # Account for the channel_axis
        if self.channel_axis is not None:
            slices_int_with_channel = (
                slices_int[: self.channel_axis]
                + (slice(None),)
                + slices_int[self.channel_axis :]
            )
        else:
            slices_int_with_channel = slices_int

        try:
            self.data[slices_int_with_channel] = 0
        except:
            raise RuntimeError(
                "Data re-initialization in the provided domain failed. Did you pass a domain range outside of the object's domain?"
            )
