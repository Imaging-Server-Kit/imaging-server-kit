from typing import Dict, Optional, Type
from .layer import Layer
from ._image import Image
from ._mask import Mask
from ._points import Points
from ._vectors import Vectors
from ._boxes import Boxes
from ._paths import Paths
from ._tracks import Tracks
from ._float import Float
from ._integer import Integer
from ._bool import Bool
from ._string import String
from ._choice import Choice
from ._notification import Notification
from ._null import Null
from ._progress import Progress


DATA_TYPES: Dict[str, Type[Layer]] = {
    c.kind: c
    for c in [
        Image,
        Mask,
        Points,
        Vectors,
        Boxes,
        Paths,
        Tracks,
        Float,
        Integer,
        Bool,
        String,
        Choice,
        Notification,
        Null,
        Progress,
    ]
}


def layer_factory(kind: str, **kwargs) -> Layer:
    """Create a data layer by passing its `kind` and initialization parameters."""
    cls: Optional[Type[Layer]] = DATA_TYPES.get(kind)
    if cls is None:
        raise ValueError(f"`{kind}` layers cannot be handled.")

    return cls(**kwargs)


__all__ = [
    "DATA_TYPES",
    "Layer",
    "Image",
    "Mask",
    "Points",
    "Vectors",
    "Boxes",
    "Paths",
    "Tracks",
    "Float",
    "Integer",
    "Bool",
    "String",
    "Choice",
    "Notification",
    "Null",
    "Progress",
]
