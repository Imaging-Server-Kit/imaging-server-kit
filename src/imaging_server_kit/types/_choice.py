from typing import List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from imaging_server_kit.types.data_layer import DataLayer


class Choice(DataLayer):
    """Data layer used to represent a choice of `items`. Can be used to represent labels for classification.

    The available choices are rendered as a dropdown selector in user interfaces.

    Example:
        choices = sk.Choice(items=["reflect", "constant"], default="reflect")
    """

    kind = "choice"
    type = str

    def __init__(
        self,
        data: Optional[str] = None,
        name="Choice",
        description="Dropdown selection",
        items: Optional[List] = None,
        required: bool = True,
        default: str = "",
        auto_call: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            data=data,
            required=required,
            default=default,
            auto_call=auto_call,
            **kwargs,
        )
        if items is None:
            items = []
        
        # Special: type defined here because it depends on items...
        self.type = Literal.__getitem__(tuple(items)) # type: ignore
