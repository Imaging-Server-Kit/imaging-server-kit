from __future__ import annotations

from typing import Dict, Optional

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer

from tqdm import tqdm


# We use a global progress bar instead of one attached to the instance
# so that it doesnt re-print itself line-by-line in the terminal.
# However, this means we can only control a single progress bar.
PBAR = tqdm()


class Progress(DataLayer):
    """Data layer used to represent a progress bar.

    Usage example:
    
    max_val = 10
    for k in range(max_val):
        yield sk.Progress(k, max_val=max_val)
    """

    kind = "progress"
    type = int

    def __init__(
        self,
        data: Optional[int] = None,
        max_val: Optional[int] = 1,
        name="Progress",
        description="Progress bar",
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):        
        if data is None:
            data = 0
        
        super().__init__(
            name=name,
            data=data,
            meta=meta,
            tile_meta=tile_meta,
            description=description,
            max_val=max_val,
            **kwargs,
        )

    def __str__(self) -> str:
        max_val = self.meta.get("max_val", 1)
        return f"Progress (current: {self.data}/{max_val})"

    def refresh(self):
        max_val = self.meta.get("max_val", 1)
        # Only print the progress bar if there is more than 1 step.
        if (max_val > 1) & (self.data is not None):
            PBAR.total = max_val
            PBAR.n = self.data
            PBAR.refresh()