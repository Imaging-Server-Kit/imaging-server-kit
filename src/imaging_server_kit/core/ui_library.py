from typing import List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class ParameterUI:
    def __init__(self, title: str = "", description: str = ""):
        self.title = title
        self.description = description
        self.type = str
        self.widget_type = None


class DropDownUI(ParameterUI):
    def __init__(
        self,
        title="Choice",
        description="Dropdown selection",
        items: List = [],
        default: str = None,
        auto_call: bool = False,
    ):
        super().__init__(title, description)
        self.type = Literal.__getitem__(tuple(items))
        self.default = default
        self.widget_type = "dropdown"
        self.auto_call = auto_call


class FloatUI(ParameterUI):
    def __init__(
        self,
        title="Float",
        description="Numeric parameter (floating point)",
        min: float = 0.0,
        max: float = 1000.0,
        step: float = 0.1,
        default: float = 0.0,
        auto_call: bool = False,
    ):
        super().__init__(title, description)
        self.min = min
        self.max = max
        self.step = step
        self.type = float
        self.default = default
        self.widget_type = "float"
        self.auto_call = auto_call


class IntUI(ParameterUI):
    def __init__(
        self,
        title="Int",
        description="Numeric parameter (integer)",
        min: int = 0,
        max: int = 1000,
        step: int = 1,
        default: int = 0,
        auto_call: bool = False,
    ):
        super().__init__(title, description)
        self.min = min
        self.max = max
        self.step = step
        self.type = int
        self.default = default
        self.widget_type = "int"
        self.auto_call = auto_call


class BoolUI(ParameterUI):
    def __init__(
        self,
        title="Bool",
        description="Boolean parameter",
        default: bool = False,
        auto_call: bool = False,
    ):
        super().__init__(title, description)
        self.type = bool
        self.default = default
        self.widget_type = "bool"
        self.auto_call = auto_call


class StringUI(ParameterUI):
    def __init__(
        self,
        title="String",
        description="String parameter",
        default: str = "",
    ):
        super().__init__(title, description)
        self.type = str
        self.default = default
        self.widget_type = "str"


### For the future: maybe we could explicitate the dimensions order?
# class nDImageUI(ParameterUI):
#     def __init__(
#         self,
#         title="nD Image",
#         description="Input image",
#         dims_order: str = "TCZYX",
#         dimensionality: List[int] = [2, 3],
#     ):
#         super().__init__(title, description)
#         self.widget_type = "image"
#         self.dims_order = dims_order
#         self.dimensionality = dimensionality


class ImageUI(ParameterUI):
    def __init__(
        self,
        title="Image",
        description="Input image (2D, 3D)",
        dimensionality: List[int] = [2, 3],
        rgb: bool = False,
    ):
        super().__init__(title, description)
        self.widget_type = "image"
        self.dimensionality = dimensionality
        self.rgb = rgb


class MaskUI(ParameterUI):
    def __init__(
        self,
        title="Mask",
        description="Segmentation mask (2D, 3D)",
        dimensionality: List[int] = [2, 3],
    ):
        super().__init__(title, description)
        self.widget_type = "mask"
        self.dimensionality = dimensionality


class PointsUI(ParameterUI):
    def __init__(
        self,
        title="Points",
        description="Input points (2D, 3D)",
        dimensionality: List[int] = [2, 3],
    ):
        super().__init__(title, description)
        self.widget_type = "points"
        self.dimensionality = dimensionality


class VectorsUI(ParameterUI):
    def __init__(
        self,
        title="Vectors",
        description="Input vectors (2D, 3D)",
        dimensionality: List[int] = [2, 3],
    ):
        super().__init__(title, description)
        self.widget_type = "vectors"
        self.dimensionality = dimensionality


class ShapesUI(ParameterUI):
    def __init__(
        self,
        title="Shapes",
        description="Input shapes",
    ):
        super().__init__(title, description)
        self.widget_type = "shapes"


class TracksUI(ParameterUI):
    def __init__(
        self,
        title="Tracks",
        description="Input tracks",
    ):
        super().__init__(title, description)
        self.widget_type = "tracks"
