from __future__ import annotations

from typing import List, Optional
import numpy as np

from imaging_server_kit.types._array import _ArrayLayer


class Image(_ArrayLayer):
    """Data layer for images: 2D or 3D arrays, optionally multichannel or RGB.

    Parameters
    ----------
    data : numpy.ndarray, optional
        The image array.
    name : str, default="Image"
        Name of the layer.
    description : str, default="Input image"
        Description of the layer, displayed on the algorithm documentation page.
    dimensionality : list of int, optional
        Accepted numbers of dimensions, for example `[2, 3]`. By default, any number
        of dimensions is accepted.
    rgb : bool, default=False
        Set to `True` for RGB images.
    channel_axis : int, optional
        Index of the channel axis, for multichannel images. The channel axis is not
        counted as a spatial dimension (it doesn't affect `ndim` and `extent`), and
        it is never split into tiles. With `rgb=True`, the last axis is always the
        channel axis.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).

    Examples
    --------
    >>> image = sk.Image(np.zeros((100, 200)), name="Blurred", colormap="viridis")
    """

    kind = "image"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Image",
        description="Input image",
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

    def _summary(self) -> str:
        summary = super()._summary()
        if self.data is None:
            return summary
        if self.meta.get("rgb"):
            summary += ", rgb"
        elif self.meta.get("channel_axis") is not None:
            summary += f", channel_axis={self.meta['channel_axis']}"
        return summary

    @property
    def channel_axis(self) -> Optional[int]:
        if self.meta:
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
