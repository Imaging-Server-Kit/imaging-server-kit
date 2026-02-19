from typing import Dict, Type
from .data_layer import DataLayer, merge_layers
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

DATA_TYPES: Dict[str, Type[DataLayer]] = {
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

__all__ = [
    "DATA_TYPES",
    "DataLayer",
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
