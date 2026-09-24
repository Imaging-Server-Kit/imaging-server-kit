from __future__ import annotations

from typing import Optional

from imaging_server_kit.core import _progress_display as progress_display
from imaging_server_kit.types.layer import Layer


class Progress(Layer):
    """Data layer used to render a progress bar in user interfaces.

    Parameters
    ----------
    data: Number of completed steps.
    max_val: Total number of steps.

    Examples
    --------
    >>> max_val = 10
    >>> for k in range(max_val):
    ...     yield sk.Progress(k, max_val=max_val)
    """

    kind = "progress"
    type = Optional[int]

    def __init__(
        self,
        data: Optional[int] = None,
        max_val: Optional[int] = 1,
        name="Progress",
        description="Progress bar",
        **kwargs,
    ):
        if data is None:
            data = 0

        super().__init__(
            name=name,
            data=data,
            description=description,
            max_val=max_val,
            **kwargs,
        )

    def __str__(self) -> str:
        max_val = self.meta.get("max_val", 1)
        return f"Progress (current: {self.data}/{max_val})"

    def _refresh(self):
        max_val = self.meta.get("max_val", 1)
        # Only show the progress bar if there is more than 1 step.
        if (max_val > 1) and (self.data is not None):
            progress_display.update(self.name, completed=self.data, total=max_val)
