from .data_layer import DataLayer
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
from ._dropdown import DropDown
from ._notification import Notification

DATA_TYPES = {
    "image": Image,
    "mask": Mask,
    "instance_mask": Mask,  # TODO
    "points": Points,
    "vectors": Vectors,
    "boxes": Boxes,
    "paths": Paths,
    "tracks": Tracks,
    "float": Float,
    "int": Integer,
    "bool": Bool,
    "str": String,
    "dropdown": DropDown,
    "notification": Notification,
}