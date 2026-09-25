from typing import Any, Optional
from imaging_server_kit.types.layer import Layer


class Null(Layer):
    """Data layer used to represent None or the absence of data.

    Parameters
    ----------
    data: Always None; accepted for interface consistency with other layers.
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
