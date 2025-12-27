"""
nD-Tiling module for the Imaging Server Kit.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class TilingError(Exception):
    def __init__(
        self,
        pixel_domain,
        provided_shape,
        message="Error during tiling occured. Provided shape (overlap or tile) is inconsistent with the pixel domain: ",
    ):
        self.message = message + f"{pixel_domain=}, {provided_shape=}"
        super().__init__(self.message)


class TileMeta:
    def __init__(self, tile_idx: int, n_tiles: int, tile_info: Dict) -> None:
        self.tile_idx = tile_idx
        self.n_tiles = n_tiles
        self.tile_info = tile_info
    
    @property
    def ndim(self) -> int:
        return self.tile_info["ndim"]

    @property
    def shape(self) -> Tuple:
        return tuple([self.tile_info[f"tile_size_{idx}"] for idx in range(self.ndim)])

    @property
    def coords_min(self) -> Tuple:
        return tuple([self.tile_info[f"pos_{idx}"] for idx in range(self.ndim)])

    @property
    def coords_max(self) -> Tuple:
        return tuple(
            [
                tile_pos + tile_size
                for (tile_pos, tile_size) in zip(self.coords_min, self.shape)
            ]
        )
    
    @property
    def overlap_px(self) -> Tuple:
        return tuple(
            [
                self.tile_info[f"overlap_px_{idx}"]
                for idx in range(self.tile_info["ndim"])
            ]
        )

    @property
    def slices(self) -> Tuple:
        return tuple(
            [
                slice(pos, max_pos)
                for pos, max_pos in zip(self.coords_min, self.coords_max)
            ]
        )
    
    @property
    def overlap_border_mask(self) -> np.ndarray:
        """Returns a boolean array selecting the rectangular region overalpping with other tiles."""
        overlap_slices = tuple(
            [slice(pos, max_pos - pos) for pos, max_pos in zip(self.overlap_px, self.shape)]
        )
        mask = np.ones(self.shape)
        mask[overlap_slices] = 0
        return mask == 1

    @property
    def overlap_count_map(self) -> np.ndarray:
        """Return an array of the same shape as the tile containing the number of overlapping tiles at each pixel."""
        overlaps_px = tuple([self.tile_info[f"overlap_px_{idx}"] for idx in range(self.ndim)])
        tile_shape = tuple([self.tile_info[f"tile_size_{idx}"] for idx in range(self.ndim)])
        is_first_tile = tuple([self.tile_info[f"first_tile_{idx}"] for idx in range(self.ndim)])
        is_last_tile = tuple([self.tile_info[f"last_tile_{idx}"] for idx in range(self.ndim)])

        per_axis = []
        for n, ov, first_tile, last_tile in zip(
            tile_shape, overlaps_px, is_first_tile, is_last_tile
        ):
            i = np.arange(n)
            c = 1
            if not first_tile:
                c = c + (i < ov).astype(np.int16)
            if not last_tile:
                c = c + (i >= n - ov).astype(np.int16)
            # TODO: check this
            per_axis.append(
                c.reshape((1,) * len(per_axis) + (n,) + (1,) * (self.ndim - len(per_axis) - 1))
            )

        overlap_count_arr = np.ones(tile_shape, dtype=np.int16)
        for c in per_axis:
            overlap_count_arr *= c

        return overlap_count_arr

    @property
    def is_first_tile(self) -> bool:
        return self.tile_idx == 0

    @property
    def is_last_tile(self) -> bool:
        return self.tile_idx == self.n_tiles - 1

    @property
    def pixel_domain(self) -> Optional[List[int]]:
        ndim = self.tile_info.get("ndim")
        if ndim is not None:
            return [self.tile_info[f"domain_size_{idx}"] for idx in range(ndim)]
    
    def serialize(self) -> Dict:
        return self.tile_info | {
            "tile_idx": self.tile_idx,
            "n_tiles": self.n_tiles,
        }


def valid_tile_size(
    tile_size_px: Union[int, Tuple, List], pixel_domain: Union[Tuple, List]
) -> bool:
    """Validates that the tile size is positive and compatible with the pixel domain."""
    if isinstance(tile_size_px, int):
        if tile_size_px <= 0:
            return False
    elif isinstance(tile_size_px, (List, Tuple)):
        if len(tile_size_px) != len(pixel_domain):
            return False
        for t in tile_size_px:
            if t <= 0:
                return False
    return True


def valid_overlap(
    overlap_percent: Union[float, Tuple, List], pixel_domain: Union[Tuple, List]
) -> bool:
    """Validate that the overlap is in the range [0-1] and compatible with the pixel domain."""
    if isinstance(overlap_percent, float):
        if (overlap_percent < 0.0) or (overlap_percent > 1.0):
            return False
    elif isinstance(overlap_percent, (Tuple, List)):
        if len(overlap_percent) != len(pixel_domain):
            return False
        for o in overlap_percent:
            if (o < 0.0) or (o > 1):
                return False
    return True


def generate_nd_tiles(
    pixel_domain: Union[Tuple, List],
    tile_size_px: Union[int, Tuple, List] = 64,
    overlap_percent: Union[float, Tuple, List] = 0.0,
    randomize: bool = False,
    delay_sec: float = 0.0,
):
    """Generate tile metadata for overlapping N-dimensional tiles over a pixel domain.

    Parameters
    ----------
    pixel_domain : tuple or list
        The size of the domain to partition into tiles, in pixels (typically, `image.shape` can be used).
    tile_size_px : int or tuple or list (optional)
        The size of an individual tile, in pixels. If an integer is provided, the same size is used
        for all dimensions. If a tuple or list is provided, it must match the dimensionality of `pixel_domain`.
        Default is 64.
    overlap_percent : float or tuple or list (optional)
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
    if delay_sec < 0:
        raise ValueError("Time delay (delay_sec) should be positive.")

    if not valid_overlap(overlap_percent, pixel_domain):
        raise ValueError(
            f"Invalid tile overlap: {overlap_percent}. Expected a value or list/tuple in the range [0-1] compatible with the pixel domain ({pixel_domain})."
        )

    if not valid_tile_size(tile_size_px, pixel_domain):
        raise ValueError(
            f"Invalid tile size: {tile_size_px}. Expected a positive integer, or list/tuple of positive integers compatible with the pixel domain ({pixel_domain})."
        )

    tiles_info = _get_tiles_info(pixel_domain, tile_size_px, overlap_percent, randomize)
    n_tiles = len(tiles_info)
    for tile_idx, tile_info in enumerate(tiles_info):
        yield TileMeta(
            tile_idx=tile_idx,
            n_tiles=n_tiles,
            tile_info=tile_info,
        )
        time.sleep(delay_sec)


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
    """Compute the tile position and size and remove parts of border tiles outside of the pixel domain."""
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


def _get_tiles_info(
    pixel_domain: Union[Tuple, List],
    tile_size_px: Union[int, Tuple, List],
    overlap_percent: Union[float, Tuple, List],
    randomize: bool,
) -> List[Dict]:
    """Tiling in N-dimensions."""
    ndim = len(pixel_domain)

    # Resolve the tile shape and overlap
    if isinstance(tile_size_px, (list, tuple)):
        tile_shape = tile_size_px
        if len(tile_shape) != ndim:
            raise TilingError(pixel_domain=pixel_domain, provided_shape=tile_shape)
    else:
        tile_shape = [tile_size_px] * ndim  # Isotropic tile

    if isinstance(overlap_percent, (list, tuple)):
        overlap = overlap_percent
        if len(overlap) != ndim:
            raise TilingError(pixel_domain=pixel_domain, provided_shape=overlap)
    else:
        overlap = [overlap_percent] * ndim

    # Overlap in pixels (TODO: we could allow users to pass an overlap in pixels directly)
    overlap_px = [
        np.round(overlap[axis] * tile_shape[axis], decimals=0).astype(int)
        for axis in range(ndim)
    ]

    tile_infos_axes = [
        _tiles_info_axis(pixel_domain[axis], tile_shape[axis], overlap_px[axis])
        for axis in range(ndim)
    ]
    n_pix_ax = [returns[0] for returns in tile_infos_axes]
    n_tiles_ax = [returns[1] for returns in tile_infos_axes]
    shift_ax = [returns[2] for returns in tile_infos_axes]
    half_pad_ax = [returns[3] for returns in tile_infos_axes]

    grid = np.meshgrid(*[np.arange(n) for n in n_tiles_ax], indexing="ij")
    tile_coords_idx = np.stack([g.ravel() for g in grid], axis=1)

    if randomize:
        np.random.shuffle(tile_coords_idx)

    tile_infos = []
    for tile_coord_ax in tile_coords_idx:
        tile_info = {"ndim": ndim} | {
            f"domain_size_{axis}": int(n_pix) for (axis, n_pix) in enumerate(n_pix_ax)
        }
        for axis, (coord_i, n_pix_i, shift_i, half_pad_i) in enumerate(
            zip(tile_coord_ax, n_pix_ax, shift_ax, half_pad_ax)
        ):
            tile_shape_i = tile_shape[axis]

            pos_i, tile_size_i, first_tile_i, last_tile_i = _get_tile_pos_and_size_i(
                coord_i,
                n_pix_i,
                shift_i,
                half_pad_i,
                tile_shape_i,
            )

            tile_info = tile_info | {
                f"pos_{axis}": int(pos_i),
                f"tile_size_{axis}": int(tile_size_i),
                f"overlap_px_{axis}": int(tile_shape_i - shift_i),
                f"first_tile_{axis}": first_tile_i,
                f"last_tile_{axis}": last_tile_i,
            }

        tile_infos.append(tile_info)

    return tile_infos
