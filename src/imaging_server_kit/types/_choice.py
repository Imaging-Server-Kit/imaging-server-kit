from typing import Dict, List, Optional

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

    def __init__(
        self,
        data: Optional[str] = None,
        name="Choice",
        description="Dropdown selection",
        items: Optional[List] = None,
        default: Optional[str] = None,
        auto_call: bool = False,
        meta: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
        )
        if items is None:
            items = []
        # Special: type defined here because it depends on items...
        self.type = Literal.__getitem__(tuple(items)) # type: ignore
        self.default = default
        self.auto_call = auto_call
        
        # Schema contributions
        main = {"default": self.default}
        extra = {"auto_call": self.auto_call}
        self.constraints = [main, extra]
        
        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

    @classmethod
    def serialize(cls, data: Optional[str], client_origin: str):
        if data is not None:
            return str(data)

    @classmethod
    def deserialize(cls, serialized_data: Optional[str], client_origin: str):
        if serialized_data is not None:
            return str(serialized_data)
