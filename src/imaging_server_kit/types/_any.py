import typing
from typing import Optional

from imaging_server_kit.types.layer import Layer


class Any(Layer):
    """Data layer for any other kind of data, such as custom Python objects.

    Parameters with types that cannot be resolved, and outputs of unrecognized types,
    are represented by this layer. They are not shown in user interfaces.

    Since this layer can hold arbitrary objects, it is generally *not* serializable,
    and only works locally (not with servers). Exceptions are simple JSON-compatible
    types (`int`, `float`, `bool`, `str`, `numpy.ndarray`), and lists, tuples, or
    dictionaries of these types.

    Parameters
    ----------
    data : object, optional
        Any kind of data.
    name : str, default="Any"
        Name of the layer.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).
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
