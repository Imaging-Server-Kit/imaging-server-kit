# Changelog

All notable, user-facing changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). This
project does not yet follow strict semantic versioning — as a young project, breaking
changes may land in minor version bumps; check this file when upgrading.

## [Unreleased]

This release introduces several breaking changes; see the migration notes below.

### Changed

- **Breaking:** Renamed several classes and methods for clarity:

  | Old (`<0.2.0`) | New (`0.2.0`) |
  | -------- | --------- |
  | `sk.Results` | `sk.Stack` |
  | `sk.DataLayer` | `sk.Layer` |
  | `sk.Results().create()` | `sk.Stack().add()` |
  | `tile_size_px` | `tile_size` |
  | `overlap_percent` | `tile_overlap` |
  | `randomize` | `tile_randomize` |
  | `delay_sec` | `tile_delay` |

- Passing `meta={...}` to a data layer is no longer required. Layer properties can now be
  passed directly as keyword arguments instead, improving readability:

  ```python
  # Old
  layer = sk.Notification("Hi", meta={"level": "info"})
  # New
  layer = sk.Notification("Hi", level="info")
  ```

- **Breaking:** Napari-related functionality (`to_napari()`, dock widgets, etc.), formerly
  distributed as the separate `napari-serverkit` package, is now bundled directly into
  `imaging-server-kit` as an optional dependency. This keeps the Napari-facing code in
  sync with the core package and simplifies installation and maintenance:

  | Old (`<0.2.0`) | New (`0.2.0`) |
  | -------- | --------- |
  | `pip install imaging-server-kit napari-serverkit` | `pip install imaging-server-kit[napari]` |

- The bridge to QuPath now goes through [QuBaLab](https://pypi.org/project/qubalab/)
  (`pip install imaging-server-kit[qupath]`, then `serverkit qupath`), replacing the
  `qupath-extension-serverkit` Java extension. The new interface offers similar
  functionality while being pure Python and easier to maintain — it also directly
  supports running algorithms progressively over large QuPath images via `tiled` mode.
  See the [documentation](https://imaging-server-kit.github.io/imaging-server-kit/) for
  details.

### Added

- `sk.Progress`, a new data layer for reporting progress through a multi-step
  computation, rendered as a progress bar in the terminal or in Napari:

  ```python
  @sk.algorithm
  def algo():
      N = 10
      for n in range(N):
          yield sk.Progress(n, max_val=N)
  ```

- A bundled set of common image analysis tools (`sk.tools`), runnable via
  `sk.to_napari(sk.tools)` / `serverkit tools napari`, or served via
  `sk.serve(sk.tools)` / `serverkit tools serve`. Includes filters (Gaussian, median,
  Sobel, variance, blobness), thresholding (manual, Otsu), morphological operators,
  connected-component labeling, object filtering (remove small objects, keep N biggest),
  hole filling, image/mask arithmetic, masking, cropping, and rescaling.

- Experimental features (may still change):
  - **`channel_axis`**: `sk.Image` and `sk.Mask` accept a `channel_axis` parameter so
    multichannel data is tiled correctly (the channel axis is excluded from tiling and
    from `position`/`size`/`ndim`, but not from `shape`).
  - **`position`**: layers now carry a `position` attribute in global coordinates, so
    several layers placed at different positions can be combined into one `sk.Stack`
    and displayed correctly relative to each other.
  - **`sk.Domain` and `select()`**: `sk.Domain` represents a sized region at a given
    position in global coordinates. It can be used to select a subset of a layer's or
    stack's data (`layer.select(domain=...)`), or to restrict an algorithm's computation
    to that region (`algo.run(..., domain=...)`).

  See the [documentation](https://imaging-server-kit.github.io/imaging-server-kit/) for
  runnable examples of these features.

### Removed

- **Breaking:** The `serverkit new` command and its `cookiecutter` template/dependency
  were removed, to keep the project focused on its core functionality. A standalone
  cookiecutter template may be provided separately in the future.

## Earlier releases

Changes prior to this file's introduction are not individually documented here. See the
[GitHub releases](https://github.com/Imaging-Server-Kit/imaging-server-kit/releases) and
[tags](https://github.com/Imaging-Server-Kit/imaging-server-kit/tags) for the full history.
