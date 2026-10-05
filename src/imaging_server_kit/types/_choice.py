from typing import List, Optional, get_args

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core._fmt import truncate


class Choice(Layer):
    """Data layer for a choice among `items`, shown as a dropdown in user interfaces.

    Can be used to represent labels for classification.

    Parameters
    ----------
    data : str, optional
        The selected item.
    name : str, default="Choice"
        Name of the layer.
    items : list of str, optional
        The available items.
    default : str, default=""
        Default item, used when `data` is not provided.
    auto_call : bool, default=False
        Re-run the algorithm when the selection changes in user interfaces.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).

    Examples
    --------
    >>> mode = sk.Choice(items=["reflect", "constant"], default="reflect")
    """

    kind = "choice"
    type = Optional[str]

    def __init__(
        self,
        data: Optional[str] = None,
        name="Choice",
        items: Optional[List] = None,
        required: bool = True,
        default: str = "",
        auto_call: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            data=data,
            required=required,
            default=default,
            auto_call=auto_call,
            **kwargs,
        )
        if items is None:
            items = []

        # Special: type defined here because it depends on items...
        self.type = Literal.__getitem__(tuple(items))  # type: ignore

    def _summary(self) -> str:
        summary = super()._summary()
        items = get_args(self.type)
        if items:
            summary += f" of {{{truncate(', '.join(str(i) for i in items))}}}"
        return summary
