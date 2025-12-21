"""
nD-Tiling module for the Imaging Server Kit.
"""

import time
from typing import Dict, List
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


def generate_nd_tiles(
    pixel_domain,
    tile_size_px=64,
    overlap_percent=0.0,
    randomize=False,
    delay_sec=0.0,
):
    """Yields tile metadata across the pixel domain, partitioned according to the given tile size etc.
    Important: tile_size_px can be an int or a list/tuple, in which case the tiles are anisotropic!
    """
    first_tile = True
    tiles_info = _get_tiles_info(pixel_domain, tile_size_px, overlap_percent, randomize)
    n_tiles = len(tiles_info)
    for tile_idx, tile_info in enumerate(tiles_info):
        tile_meta = {
            "tile_params": tile_info
            | {
                "tile_idx": tile_idx,
                "n_tiles": n_tiles,
            }
        }
        if first_tile:
            tile_meta["tile_params"]["first_tile"] = True
            first_tile = False
        yield tile_meta
        time.sleep(delay_sec)


def overlap_count_map(tile_info: Dict) -> np.ndarray:
    """Returns an array of the same shape as the tile containing the number of overlapping tiles at each pixel."""
    ndim = tile_info["tile_params"]["ndim"]

    overlaps_px = tuple(
        [tile_info["tile_params"][f"overlap_px_{idx}"] for idx in range(ndim)]
    )

    tile_shape = tuple(
        [tile_info["tile_params"][f"tile_size_{idx}"] for idx in range(ndim)]
    )

    is_first_tile = tuple(
        [tile_info["tile_params"][f"first_tile_{idx}"] for idx in range(ndim)]
    )

    is_last_tile = tuple(
        [tile_info["tile_params"][f"last_tile_{idx}"] for idx in range(ndim)]
    )

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
        per_axis.append(
            c.reshape((1,) * len(per_axis) + (n,) + (1,) * (ndim - len(per_axis) - 1))
        )

    overlap_count_arr = np.ones(tile_shape, dtype=np.int16)
    for c in per_axis:
        overlap_count_arr *= c

    return overlap_count_arr


def _tiles_info_axis(n_pix_i, tile_shape_i, overlap_i):
    overlap_px_i = np.round(overlap_i * tile_shape_i, decimals=0).astype(int)

    n_tiles_i = np.ceil(n_pix_i / (tile_shape_i - overlap_px_i)).astype(int)

    pad_i = n_tiles_i * (tile_shape_i - overlap_px_i) - n_pix_i

    half_pad_i = np.round(pad_i / 2, decimals=0).astype(int)

    shift_i = tile_shape_i - overlap_px_i

    return n_pix_i, n_tiles_i, shift_i, half_pad_i


def _get_tile_pos_and_size_i(coord_i_, n_pix_i, shift_i, half_pad_i, tile_shape_i):
    coord_i = coord_i_ * shift_i - half_pad_i
    if coord_i < 0:
        pos_i = 0
        di_left = -coord_i
        first_tile_i = True
    else:
        pos_i = coord_i
        di_left = 0
        first_tile_i = False
    if (coord_i + tile_shape_i) >= n_pix_i:
        di_right = coord_i + tile_shape_i - n_pix_i
        last_tile_i = True
    else:
        di_right = 0
        last_tile_i = False

    tile_size_i = tile_shape_i - di_left - di_right

    return pos_i, tile_size_i, first_tile_i, last_tile_i


def _get_tiles_info(
    pixel_domain, tile_size_px, overlap_percent, randomize
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

    tile_infos_axes = [
        _tiles_info_axis(pixel_domain[axis], tile_shape[axis], overlap[axis])
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
        valid_tile = True
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

            if pos_i >= n_pix_i:  # This sometimes happens..
                valid_tile = False
                break

        if valid_tile:
            tile_infos.append(tile_info)

    return tile_infos
