import time
import numpy as np


def _pad_image_2d(image_xyc, tile_size_px, overlap_px):
    """Pads the image so that its length and width are divisible by the tile size (which is square)."""
    rx, ry = image_xyc.shape[0], image_xyc.shape[1]

    pad_x = rx % (tile_size_px - overlap_px) + overlap_px
    half_pad_x = pad_x // 2
    nx_ceil = np.ceil((rx + pad_x - overlap_px) / (tile_size_px - overlap_px)).astype(
        int
    )
    padded_image_size_x = nx_ceil * (tile_size_px - overlap_px) + overlap_px

    pad_y = ry % (tile_size_px - overlap_px) + overlap_px
    half_pad_y = pad_y // 2
    ny_ceil = np.ceil((ry + pad_y - overlap_px) / (tile_size_px - overlap_px)).astype(
        int
    )
    padded_image_size_y = ny_ceil * (tile_size_px - overlap_px) + overlap_px

    if len(image_xyc.shape) == 2:
        # Grayscale case
        image_padded = np.zeros((padded_image_size_x, padded_image_size_y))
    elif len(image_xyc.shape) == 3:
        # RGB case
        n_channels = image_xyc.shape[2]
        image_padded = np.zeros((padded_image_size_x, padded_image_size_y, n_channels))

    image_padded[half_pad_x : (half_pad_x + rx), half_pad_y : (half_pad_y + ry)] = (
        image_xyc
    )

    return image_padded, (half_pad_x, half_pad_y), (nx_ceil, ny_ceil)


def image_tile_generator_2D(image_xy, tile_size_px, overlap_percent=0, delay=0, randomize=False):
    """Generates image tiles and their coordinates in the image domain."""
    is_rgb = image_xy.ndim == 3

    overlap_px = np.round(overlap_percent * tile_size_px, decimals=0).astype(int)
    image_p, (half_pad_x, half_pad_y), (nx, ny) = _pad_image_2d(
        image_xy,
        tile_size_px,
        overlap_px,
    )
    rx, ry = image_xy.shape[0], image_xy.shape[1]
    shift_x = tile_size_px - overlap_px
    shift_y = tile_size_px - overlap_px
    n_tiles = nx * ny
    tile_idx = 0

    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    tile_coord_idx = np.stack([xx.ravel(), yy.ravel()], axis=1)

    if randomize:
        np.random.shuffle(tile_coord_idx)

    for (ix, iy) in tile_coord_idx:
        image_tile = image_p[
            (ix * shift_x) : (ix * shift_x + tile_size_px),
            (iy * shift_y) : (iy * shift_y + tile_size_px),
        ]
        coord_x = ix * shift_x - half_pad_x
        coord_y = iy * shift_y - half_pad_y

        crop_x_start = max(0, -coord_x)
        crop_y_start = max(0, -coord_y)
        crop_x_end = tile_size_px - max(0, (coord_x + tile_size_px - rx))
        crop_y_end = tile_size_px - max(0, (coord_y + tile_size_px - ry))

        image_tile = image_tile[crop_x_start:crop_x_end, crop_y_start:crop_y_end]
        pos_x = max(0, coord_x)
        pos_y = max(0, coord_y)

        tile_meta = {
            "tile_params": {
                "pos_x": int(pos_x),
                "pos_y": int(pos_y),
                "image_size_x": int(rx),
                "image_size_y": int(ry),
                "tile_idx": int(tile_idx),
                "n_tiles": int(n_tiles),
                "rgb": is_rgb,
            }
        }

        tile_idx += 1

        yield image_tile, tile_meta
        
        time.sleep(delay)


def _pad_image_3d(image_zyx, tile_size_px, overlap_px):
    """Pads the image so that its length and width are divisible by the tile size (which is square)."""
    rz, ry, rx = image_zyx.shape[0], image_zyx.shape[1], image_zyx.shape[2]

    pad_x = rx % (tile_size_px - overlap_px) + overlap_px
    half_pad_x = pad_x // 2
    nx_ceil = np.ceil((rx + pad_x - overlap_px) / (tile_size_px - overlap_px)).astype(
        int
    )
    padded_image_size_x = nx_ceil * (tile_size_px - overlap_px) + overlap_px

    pad_y = ry % (tile_size_px - overlap_px) + overlap_px
    half_pad_y = pad_y // 2
    ny_ceil = np.ceil((ry + pad_y - overlap_px) / (tile_size_px - overlap_px)).astype(
        int
    )
    padded_image_size_y = ny_ceil * (tile_size_px - overlap_px) + overlap_px

    pad_z = rz % (tile_size_px - overlap_px) + overlap_px
    half_pad_z = pad_z // 2
    nz_ceil = np.ceil((rz + pad_z - overlap_px) / (tile_size_px - overlap_px)).astype(
        int
    )
    padded_image_size_z = nz_ceil * (tile_size_px - overlap_px) + overlap_px

    image_padded = np.zeros((padded_image_size_z, padded_image_size_y, padded_image_size_x))

    image_padded[half_pad_z : (half_pad_z + rz), half_pad_y : (half_pad_y + ry), half_pad_x : (half_pad_x + rx)] = (
        image_zyx
    )

    return image_padded, (half_pad_z, half_pad_y, half_pad_x), (nz_ceil, ny_ceil, nx_ceil)


def image_tile_generator_3D(image_zyx, tile_size_px, overlap_percent=0, delay=0, randomize=False):
    """Generates image tiles and their coordinates in the image domain."""
    overlap_px = np.round(overlap_percent * tile_size_px, decimals=0).astype(int)
    image_p, (half_pad_z, half_pad_y, half_pad_x), (nz, ny, nx) = _pad_image_3d(
        image_zyx,
        tile_size_px,
        overlap_px,
    )
    rz, ry, rx = image_zyx.shape
    shift_z = tile_size_px - overlap_px
    shift_y = tile_size_px - overlap_px
    shift_x = tile_size_px - overlap_px
    n_tiles = nx * ny * nz
    tile_idx = 0


    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
    tile_coord_idx = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)

    if randomize:
        np.random.shuffle(tile_coord_idx)

    for (iz, iy, ix) in tile_coord_idx:
    # for iz in range(nz):
    #     for iy in range(ny):
    #         for ix in range(nx):
        image_tile = image_p[
            (iz * shift_z) : (iz * shift_z + tile_size_px),
            (iy * shift_y) : (iy * shift_y + tile_size_px),
            (ix * shift_x) : (ix * shift_x + tile_size_px),
        ]
        coord_z = iz * shift_z - half_pad_z
        coord_y = iy * shift_y - half_pad_y
        coord_x = ix * shift_x - half_pad_x

        crop_z_start = max(0, -coord_z)
        crop_y_start = max(0, -coord_y)
        crop_x_start = max(0, -coord_x)

        crop_z_end = tile_size_px - max(0, (coord_z + tile_size_px - rz))
        crop_y_end = tile_size_px - max(0, (coord_y + tile_size_px - ry))
        crop_x_end = tile_size_px - max(0, (coord_x + tile_size_px - rx))

        image_tile = image_tile[crop_z_start:crop_z_end, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        pos_z = max(0, coord_z)
        pos_y = max(0, coord_y)
        pos_x = max(0, coord_x)

        tile_meta = {
            "tile_params": {
                "pos_z": int(pos_z),
                "pos_y": int(pos_y),
                "pos_x": int(pos_x),
                "image_size_z": int(rz),
                "image_size_y": int(ry),
                "image_size_x": int(rx),
                "tile_idx": int(tile_idx),
                "n_tiles": int(n_tiles),
                "rgb": False,
            }
        }

        tile_idx += 1

        yield image_tile, tile_meta
        
        time.sleep(delay)


def initialize_tiled_image(tile_params):
    if tile_params.get("image_size_z"):
        tiled_image = np.zeros(
            (
                tile_params.get("image_size_z"),
                tile_params.get("image_size_y"),
                tile_params.get("image_size_x"),
            )
        )
    else:
        if tile_params.get("rgb"):
            tiled_image = np.zeros(
                (
                    tile_params.get("image_size_x"),
                    tile_params.get("image_size_y"),
                    3,
                )
            )
        else:
            tiled_image = np.zeros(
                (
                    tile_params.get("image_size_x"),
                    tile_params.get("image_size_y"),
                )
            )
    return tiled_image