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

    Example:
        notif = sk.Progress()
    """

    kind = "progress"
    type = int

    def __init__(
        self,
        data: Optional[int] = None,
        name="Progress",
        description="Progress bar",
        default: Optional[int] = None,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
    ):
        # Initialize max_val
        if meta is None:
            meta = {"max_val": 1}
        else:
            if "max_val" not in meta:
                meta["max_val"] = 1
        
        if data is None:
            data = 0
        
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
        )
        self.default = default
        
        # Schema contributions
        main = {"default": self.default}
        extra = {}
        self.constraints = [main, extra]
        
        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

    def __str__(self) -> str:
        max_val = self.meta["max_val"]
        return f"Progress (current: {self.data}/{max_val})"

    def refresh(self):
        max_val = self.meta["max_val"]
        # Only print the progress bar if there is more than 1 step.
        if (max_val > 1) & (self.data is not None):
            PBAR.total = max_val
            PBAR.n = self.data
            PBAR.refresh()