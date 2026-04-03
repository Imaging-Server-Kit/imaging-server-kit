"""
nD-Tiling module for the Imaging Server Kit.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np

from imaging_server_kit.core.domain import Domain


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
class TilingSpecs:
    tile_size: Union[int, Tuple, List] = 64
    tile_overlap: Union[float, Tuple, List] = 0.0
    tile_order_random: bool = False
    tile_delay: float = 0.0


class TileMeta:
    def __init__(
        self,
        tile_idx: Optional[int] = None,
        n_tiles: Optional[int] = None,
        first_tile: Optional[Union[Tuple, List]] = None,
        last_tile: Optional[Union[Tuple, List]] = None,
        overlap_px: Optional[Union[Tuple, List]] = None,
    ):
        self.tile_idx = 0 if tile_idx is None else tile_idx
        self.n_tiles = 1 if n_tiles is None else n_tiles
        self._first_tile = first_tile
        self._last_tile = last_tile
        self._overlap_px = overlap_px

    def __str__(self):
        message = "Tile Meta"
        message += "\n"
        message += f"Index: {self.tile_idx}"
        message += "\n"
        message += f"Tiles: {self.n_tiles}"
        return message

    def __repr__(self):
        return self.__str__()

    @property
    def overlap_px(self) -> Optional[Tuple]:
        if self._overlap_px is not None:
            return tuple(self._overlap_px)

    @overlap_px.setter
    def overlap_px(self, value: Optional[Tuple]):
        self._overlap_px = value

    @property
    def first_tile(self) -> Optional[Tuple]:
        if self._first_tile is not None:
            return tuple(self._first_tile)

    @first_tile.setter
    def first_tile(self, value: Optional[Tuple]):
        self._first_tile = value

    @property
    def last_tile(self) -> Optional[Tuple]:
        if self._last_tile is not None:
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
    domain: Optional[Domain] = None,
    tile_size: Union[int, Tuple, List] = 64,
    tile_overlap: Union[float, Tuple, List] = 0.0,
    tile_order_random: bool = False,
    tile_delay: float = 0.0,
) -> Generator[Tuple[TileMeta, Domain], None, None]:
    """Generate tile metadata for overlapping N-dimensional tiles over a pixel domain.

    Parameters
    ----------
    bounds : optional sk.Domain
        The domain to partition into tiles.
    tile_size : int or tuple or list (optional)
        The size of an individual tile, in pixels. If an integer is provided, the same size is used
        for all dimensions. If a tuple or list is provided, it must match the dimensionality of `bounds`.
        Default is 64.
    tile_overlap : float or tuple or list (optional)
        Relative overlap between adjacent tiles (in the range [0, 1]). If an integer is provided, the same relative
        overlap is used for all dimensions.
        Default is 0.0 (no overlap).
    tile_order_random : bool, optional
        If True, randomize the order in which tiles are yielded. Default is False.
    tile_delay : float, optional
        Time delay in seconds to wait after yielding each tile. Default is 0.0 (no delay).

    Yields
    ------
    A dictionary `tile_info` that contains metadata about the generated tile, including its position, size, and index in the tile series.
    """
    if domain is None:
        yield (TileMeta(), Domain())
    else:
        if tile_delay < 0:
            raise ValueError("Time delay should be positive.")

        if not _valid_overlap(overlap=tile_overlap, bounds=domain.size):
            raise ValueError(
                f"Invalid tile overlap: {tile_overlap}. Expected a value or list/tuple in the range [0-1] compatible with the domain size ({domain.size})."
            )

        if not _valid_tile_size(tile_size=tile_size, bounds=domain.size):
            raise ValueError(
                f"Invalid tile size: {tile_size}. Expected a positive integer, or list/tuple of positive integers compatible with the domain size ({domain.size})."
            )

        ctx = TilingSpecs(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_order_random=tile_order_random,
            tile_delay=tile_delay,
        )

        for tile_meta, tile_domain in _generate_tile_meta(domain=domain, ctx=ctx):
            time.sleep(tile_delay)
            yield (tile_meta, tile_domain)


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
    domain: Domain, ctx: TilingSpecs
) -> Generator[Tuple[TileMeta, Domain], None, None]:
    """Tiling in N-dimensions."""
    ndim = domain.ndim
    size = domain.size
    coords_min = domain.coords_min

    # Resolve the tile shape and overlap
    if isinstance(ctx.tile_size, (list, tuple)):
        tile_shape = ctx.tile_size
        if len(tile_shape) != ndim:
            raise TilingError(bounds=size, provided_shape=tile_shape)
    else:
        tile_shape = [ctx.tile_size] * ndim  # Isotropic tile

    if isinstance(ctx.tile_overlap, (list, tuple)):
        overlap = ctx.tile_overlap
        if len(overlap) != ndim:
            raise TilingError(bounds=size, provided_shape=overlap)
    else:
        overlap = [ctx.tile_overlap] * ndim

    # Overlap in pixels (TODO: we could allow users to pass an overlap in pixels directly)
    overlap_px = [
        np.round(overlap[axis] * tile_shape[axis], decimals=0).astype(int)
        for axis in range(ndim)
    ]

    tile_infos_axes = [
        _tiles_info_axis(size[axis], tile_shape[axis], overlap_px[axis])
        for axis in range(ndim)
    ]
    n_pix_ax = [returns[0] for returns in tile_infos_axes]
    n_tiles_ax = [returns[1] for returns in tile_infos_axes]
    shift_ax = [returns[2] for returns in tile_infos_axes]
    half_pad_ax = [returns[3] for returns in tile_infos_axes]

    grid = np.meshgrid(*[np.arange(n) for n in n_tiles_ax], indexing="ij")
    tile_coords_idx = np.stack([g.ravel() for g in grid], axis=1)

    if ctx.tile_order_random:
        np.random.shuffle(tile_coords_idx)

    n_tiles = len(tile_coords_idx)
    if n_tiles == 0:
        yield (TileMeta(), Domain())
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
            )

            position = [tile_pos[i] + coords_min[i] for i in range(ndim)]

            tile_domain = Domain(
                size=tile_size,
                position=position,
            )

            yield (tile_meta, tile_domain)
