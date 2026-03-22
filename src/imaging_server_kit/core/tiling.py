"""
nD-Tiling module for the Imaging Server Kit.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np


class TilingError(Exception):
    def __init__(
        self,
        bounds,
        provided_shape,
        message="Error during tiling occured. Provided shape (overlap or tile) is inconsistent with the pixel bounds: ",
    ):
        self.message = message + f"{bounds=}, {provided_shape=}"
        super().__init__(self.message)


@dataclass
class TilingContext:
    tile_size: Union[int, Tuple, List] = 64
    overlap: Union[float, Tuple, List] = 0.0
    randomize: bool = False
    delay_sec: float = 0.0


class TileMeta:
    def __init__(
        self,
        tile_idx: Optional[int] = None,
        n_tiles: Optional[int] = None,
        first_tile: Optional[Union[Tuple, List]] = None,
        last_tile: Optional[Union[Tuple, List]] = None,
        overlap_px: Optional[Union[Tuple, List]] = None,
        tile_size: Optional[Union[Tuple, List]] = None,
        tile_pos: Optional[Union[Tuple, List]] = None,
    ):
        self.tile_idx = 0 if tile_idx is None else tile_idx
        self.n_tiles = 1 if n_tiles is None else n_tiles
        self._first_tile = first_tile
        self._last_tile = last_tile
        self._overlap_px = overlap_px
        self._tile_size = tile_size
        self._coords_min = tile_pos

    def __str__(self):
        message = "TileMeta"
        message += "\n"
        message += f"coords_min: {self.coords_min}"
        message += "\n"
        message += f"tile_size: {self.tile_size}"
        return message

    def __repr__(self):
        return self.__str__()

    @property
    def tile_size(self) -> Optional[Tuple]:
        if self._tile_size is not None:
            return tuple(self._tile_size)

    @tile_size.setter
    def tile_size(self, value: Optional[Tuple]):
        self._tile_size = value

    @property
    def coords_min(self) -> Optional[Tuple]:
        if self._coords_min is not None:
            return tuple(self._coords_min)

    @coords_min.setter
    def coords_min(self, value: Optional[Tuple]):
        self._coords_min = value

    @property
    def coords_max(self) -> Optional[Tuple]:
        if (self.coords_min is None) or (self.tile_size is None):
            return

        return tuple(
            [
                tile_pos + tile_size
                for (tile_pos, tile_size) in zip(self.coords_min, self.tile_size)
            ]
        )

    @property
    def ndim(self) -> Optional[int]:
        if self.coords_max is not None:
            return len(self.coords_max)

    @property
    def overlap_px(self) -> Optional[Tuple]:
        if self._overlap_px is not None:
            return tuple(self._overlap_px)

    @overlap_px.setter
    def overlap_px(self, value: Optional[Tuple]):
        self._overlap_px = value

    @property
    def slices(self) -> Optional[Tuple]:
        if (self.coords_min is None) or (self.coords_max is None):
            return

        return tuple(
            [
                slice(pos, max_pos)
                for pos, max_pos in zip(self.coords_min, self.coords_max)
            ]
        )

    @property
    def overlap_border_mask(self) -> Optional[np.ndarray]:
        """Returns a boolean array selecting the rectangular region overalpping with other tiles."""
        if (self.overlap_px is None) or (self.tile_size is None):
            return

        overlap_slices = tuple(
            [
                slice(pos, max_pos - pos)
                for pos, max_pos in zip(self.overlap_px, self.tile_size)
            ]
        )
        mask = np.ones(self.tile_size)
        mask[overlap_slices] = 0
        return mask == 1

    @property
    def overlap_count_map(self) -> Optional[np.ndarray]:
        """Return an array of the same shape as the tile containing the number of overlapping tiles at each pixel."""
        if (
            (self.first_tile is None)
            or (self.last_tile is None)
            or (self.tile_size is None)
            or (self.overlap_px is None)
            or (self.ndim is None)
        ):
            return

        per_axis = []
        for n, ov, first_tile, last_tile in zip(
            self.tile_size, self.overlap_px, self.first_tile, self.last_tile
        ):
            i = np.arange(n)
            c = np.ones(n, dtype=np.int16)
            if not first_tile:
                c = c + (i < ov).astype(np.int16)
            if not last_tile:
                c = c + (i >= n - ov).astype(np.int16)
            per_axis.append(
                c.reshape(
                    (1,) * len(per_axis) + (n,) + (1,) * (self.ndim - len(per_axis) - 1)
                )
            )

        overlap_count_arr = np.ones(self.tile_size, dtype=np.int16)
        for c in per_axis:
            overlap_count_arr *= c.reshape(
                c.shape
                + (1,)
                * (
                    overlap_count_arr.ndim - c.ndim
                )  # Add fake dims (fixes the RGB case)
            )

        return overlap_count_arr

    @property
    def first_tile(self) -> Optional[Tuple]:
        if self._first_tile is None:
            if self.ndim is not None:
                return tuple([False] * self.ndim)
        else:
            return tuple(self._first_tile)

    @first_tile.setter
    def first_tile(self, value: Optional[Tuple]):
        self._first_tile = value

    @property
    def last_tile(self) -> Optional[Tuple]:
        if self._last_tile is None:
            if self.ndim is not None:
                return tuple([False] * self.ndim)
        else:
            return tuple(self._last_tile)

    @last_tile.setter
    def last_tile(self, value: Optional[Tuple]):
        self._last_tile = value

    @property
    def is_first_tile(self) -> bool:
        if self.tile_idx is not None:
            return self.tile_idx == 0
        return False

    @property
    def is_last_tile(self) -> bool:
        if self.n_tiles is not None:
            return self.tile_idx == self.n_tiles - 1
        return False

    def serialize(self) -> Dict:
        return {
            "tile_idx": self.tile_idx,
            "n_tiles": self.n_tiles,
            "first_tile": self.first_tile,
            "last_tile": self.last_tile,
            "overlap_px": self.overlap_px,
            "tile_size": self._tile_size,
            "tile_pos": self._coords_min,
        }

    def copy(self) -> TileMeta:
        return TileMeta(**self.serialize())


def _valid_tile_size(
    tile_size: Union[int, Tuple, List], bounds: Union[Tuple, List]
) -> bool:
    """Validates that the tile size is positive and compatible with the pixel bounds."""
    if isinstance(tile_size, int):
        if tile_size <= 0:
            return False
    elif isinstance(tile_size, (List, Tuple)):
        if len(tile_size) != len(bounds):
            return False
        for t in tile_size:
            if t <= 0:
                return False
    return True


def _valid_overlap(
    overlap: Union[float, Tuple, List], bounds: Union[Tuple, List]
) -> bool:
    """Validate that the overlap is in the range [0-1] and compatible with the pixel bounds."""
    if isinstance(overlap, float):
        if (overlap < 0.0) or (overlap > 1.0):
            return False
    elif isinstance(overlap, (Tuple, List)):
        if len(overlap) != len(bounds):
            return False
        for o in overlap:
            if (o < 0.0) or (o > 1):
                return False
    return True


def generate_tiles(
    bounds: Optional[Union[Tuple, List]] = None,
    tile_size: Union[int, Tuple, List] = 64,
    overlap: Union[float, Tuple, List] = 0.0,
    randomize: bool = False,
    delay_sec: float = 0.0,
) -> Generator[TileMeta, None, None]:
    """Generate tile metadata for overlapping N-dimensional tiles over a pixel domain.

    Parameters
    ----------
    bounds : tuple or list
        The size of the domain to partition into tiles, in pixels (typically, `image.shape` can be used).
    tile_size : int or tuple or list (optional)
        The size of an individual tile, in pixels. If an integer is provided, the same size is used
        for all dimensions. If a tuple or list is provided, it must match the dimensionality of `bounds`.
        Default is 64.
    overlap : float or tuple or list (optional)
        Relative overlap between adjacent tiles (in the range [0, 1]). If an integer is provided, the same relative
        overlap is used for all dimensions.
        Default is 0.0 (no overlap).
    randomize : bool, optional
        If True, randomize the order in which tiles are yielded. Default is False.
    delay_sec : float, optional
        Time delay in seconds to wait after yielding each tile. Default is 0.0 (no delay).

    Yields
    ------
    A dictionary `tile_info` that contains metadata about the generated tile, including its position, size, and index in the tile series.
    """
    if bounds is None:
        yield TileMeta()
    else:
        if delay_sec < 0:
            raise ValueError("Time delay (delay_sec) should be positive.")

        if not _valid_overlap(overlap=overlap, bounds=bounds):
            raise ValueError(
                f"Invalid tile overlap: {overlap}. Expected a value or list/tuple in the range [0-1] compatible with the pixel domain ({bounds})."
            )

        if not _valid_tile_size(tile_size=tile_size, bounds=bounds):
            raise ValueError(
                f"Invalid tile size: {tile_size}. Expected a positive integer, or list/tuple of positive integers compatible with the pixel domain ({bounds})."
            )

        ctx = TilingContext(
            tile_size=tile_size,
            overlap=overlap,
            randomize=randomize,
            delay_sec=delay_sec,
        )

        for tile_meta in _generate_tile_meta(bounds=bounds, ctx=ctx):
            time.sleep(delay_sec)
            yield tile_meta


def _tiles_info_axis(size_i: int, tile_size_i: int, overlap_px_i: int):
    """Compute tiling variables for a given axis `i`."""
    # Number of tiles along the axis
    n_tiles_i = np.ceil(size_i / (tile_size_i - overlap_px_i)).astype(int)

    # Extra pixels to pad along the axis
    pad_i = n_tiles_i * (tile_size_i - overlap_px_i) - size_i

    # Number of pixels to pad on each side of the axis
    half_pad_i = np.round(pad_i / 2, decimals=0).astype(int)

    # Amount of translation betwen consecutive tiles
    shift_i = tile_size_i - overlap_px_i

    return size_i, n_tiles_i, shift_i, half_pad_i


def _get_tile_pos_and_size_i(
    idx_i: int, size_i: int, shift_i: int, half_pad_i: int, tile_size_i: int
):
    """Compute the tile position and size and remove parts of border tiles outside of the pixel bounds."""
    coord_i = idx_i * shift_i - half_pad_i
    if coord_i < 0:
        pos_i = 0
        trim_left = -coord_i
        is_first_tile_i = True
    else:
        pos_i = coord_i
        trim_left = 0
        is_first_tile_i = False
    if (coord_i + tile_size_i) >= size_i:
        trim_right = coord_i + tile_size_i - size_i
        is_last_tile_i = True
    else:
        trim_right = 0
        is_last_tile_i = False

    # Reduce the tile size if needed
    tile_size_i = tile_size_i - trim_left - trim_right

    return pos_i, tile_size_i, is_first_tile_i, is_last_tile_i


def _generate_tile_meta(
    bounds: Union[Tuple, List], ctx: TilingContext
) -> Generator[TileMeta, None, None]:
    """Tiling in N-dimensions."""
    ndim = len(bounds)

    # Resolve the tile shape and overlap
    if isinstance(ctx.tile_size, (list, tuple)):
        tile_shape = ctx.tile_size
        if len(tile_shape) != ndim:
            raise TilingError(bounds=bounds, provided_shape=tile_shape)
    else:
        tile_shape = [ctx.tile_size] * ndim  # Isotropic tile

    if isinstance(ctx.overlap, (list, tuple)):
        overlap = ctx.overlap
        if len(overlap) != ndim:
            raise TilingError(bounds=bounds, provided_shape=overlap)
    else:
        overlap = [ctx.overlap] * ndim

    # Overlap in pixels (TODO: we could allow users to pass an overlap in pixels directly)
    overlap_px = [
        np.round(overlap[axis] * tile_shape[axis], decimals=0).astype(int)
        for axis in range(ndim)
    ]

    tile_infos_axes = [
        _tiles_info_axis(bounds[axis], tile_shape[axis], overlap_px[axis])
        for axis in range(ndim)
    ]
    n_pix_ax = [returns[0] for returns in tile_infos_axes]
    n_tiles_ax = [returns[1] for returns in tile_infos_axes]
    shift_ax = [returns[2] for returns in tile_infos_axes]
    half_pad_ax = [returns[3] for returns in tile_infos_axes]

    grid = np.meshgrid(*[np.arange(n) for n in n_tiles_ax], indexing="ij")
    tile_coords_idx = np.stack([g.ravel() for g in grid], axis=1)

    if ctx.randomize:
        np.random.shuffle(tile_coords_idx)

    n_tiles = len(tile_coords_idx)
    if n_tiles == 0:
        yield TileMeta()
    else:
        for tile_idx, tile_coord_ax in enumerate(tile_coords_idx):
            first_tile = []
            last_tile = []
            overlap_px = []
            tile_size = []
            tile_pos = []
            for axis, (coord_i, n_pix_i, shift_i, half_pad_i) in enumerate(
                zip(tile_coord_ax, n_pix_ax, shift_ax, half_pad_ax)
            ):
                tile_shape_i = tile_shape[axis]
                pos_i, tile_size_i, first_tile_i, last_tile_i = (
                    _get_tile_pos_and_size_i(
                        coord_i,
                        n_pix_i,
                        shift_i,
                        half_pad_i,
                        tile_shape_i,
                    )
                )
                first_tile.append(first_tile_i)
                last_tile.append(last_tile_i)
                overlap_px.append(int(tile_shape_i - shift_i))
                tile_size.append(int(tile_size_i))
                tile_pos.append(int(pos_i))

            tile_meta = TileMeta(
                tile_idx=tile_idx,
                n_tiles=len(tile_coords_idx),
                first_tile=first_tile,
                last_tile=last_tile,
                overlap_px=overlap_px,
                tile_size=tile_size,
                tile_pos=tile_pos,
            )
            yield tile_meta
