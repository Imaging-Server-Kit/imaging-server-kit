from typing import Any, Optional
from imaging_server_kit.types.layer import Layer


class Null(Layer):
    """Data layer for `None`, or the absence of data.

    Parameters
    ----------
    data : None, optional
        Always `None`; accepted for consistency with other layers.
    name : str, default="None"
        Name of the layer.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).
    """

    kind = "null"
    type = type(None)

    def __init__(
        self,
        data: Optional[Any] = None,
        name="None",
        default=None,
        serializer: str = "default",
        **kwargs,
    ):
        super().__init__(
            name=name,
            data=data,
            default=default,
            serializer=serializer,
            **kwargs,
        )

    def _summary(self) -> str:
        return ""
