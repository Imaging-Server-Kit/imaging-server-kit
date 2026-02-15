from typing import Dict, List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from imaging_server_kit.core.tiling import TileMeta
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
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
            default=default,
            auto_call=auto_call,
            **kwargs,
        )
        if items is None:
            items = []
        
        # Special: type defined here because it depends on items...
        self.type = Literal.__getitem__(tuple(items)) # type: ignore
