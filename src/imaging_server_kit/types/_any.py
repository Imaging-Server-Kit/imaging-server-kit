import typing
from typing import Optional

from imaging_server_kit.types.layer import Layer


class Any(Layer):
    """Data layer used to represent "any" kind of data, such as custom Python objects.
    
    Note: Since this type is made to encapsulate arbitrary custom classes, it is generally 
    *not* serializable (only usable locally). Excepthions: if `data` is a simple json-compiant 
    type (int, float, bool, str, np.ndarray) or a list, tuple, or dictionary of these types, 
    the remote functions (which need serialization) still work.

    Parameters
    ----------
    data: Any kind of data.
    """

    kind = "any"
    type = typing.Any

    def __init__(
        self,
        data: Optional[typing.Any] = None,
        name="Any",
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
