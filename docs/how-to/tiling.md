# Run algorithms tile-by-tile

Tiled inference is used to run image processing filters, segmentation, or detection algorithms **tile-by-tile** instead of on the whole input image at once. The processed tiles are progressively assembled into the final result.

## Enabling tiling

Tiling is disabled by default. Enable it with `tileable=True` when defining the algorithm:

```python
import imaging_server_kit as sk

@sk.algorithm(tileable=True)  # <- Set tileable=True
def threshold_algo(image, threshold=128):
    mask = image > threshold
    return sk.Mask(mask)
```

Running an algorithm in tiles without `tileable=True` raises an `AlgorithmRuntimeError`.

## Tiling in Napari

```python
import skimage.data

viewer = sk.to_napari(threshold_algo)
viewer.add_image(skimage.data.coins())
```

Before running the algorithm, expand the *Tiled inference* section of the widget and check *Run in tiles*. You can also adjust the tile size, the overlap, a delay between tiles, and whether tiles are processed in a random order.

<video width="512" controls loop autoplay muted playsinline>
  <source src="../assets/videos/tiling2.mp4" type="video/mp4">
</video>

## Tiling in Python

Set `tiled=True` when calling `.run()`, and optionally specify the tiling parameters:

```python
image = skimage.data.coins()

results = threshold_algo.run(
    image,
    tiled=True,  # <- Enable tiled inference
    tile_size=64,  # (64, 64) tiles. Use e.g. (32, 64) for anisotropic tiles.
    tile_overlap=0.1,  # 10% overlap
    tile_randomize=True,  # Process the tiles in a random order
    tile_delay=0.0,  # (Optional) Add a time delay between tiles, in seconds
)
```

| Parameter | Default | Description |
|---|---|---|
| `tile_size` | `64` | Tile size in pixels. A single value, or one value per axis. |
| `tile_overlap` | `0.0` | Overlap between neighbouring tiles, relative to the tile size. |
| `tile_randomize` | `False` | Process the tiles in a random order. |
| `tile_delay` | `0.0` | Extra delay between tiles, in seconds. |

Tiling works the same way with local algorithms and with clients connected to a server:

```python
client = sk.Client("http://localhost:8000")

results = client.run(image, tiled=True, tile_size=64, tile_overlap=0.1)
```

## Instance segmentation

!!! warning "Work in progress"
    Running instance segmentation algorithms in tiles is still an experimental feature.  

By default, overlapping regions of masks are overwritten by the last tile. For **instance segmentation**, where each object has its own label, return `sk.Mask(..., merger="instances")`. Labels are then made unique across tiles, and objects crossing tile borders are stitched together. Stitching requires a non-zero `tile_overlap`.

```python
from skimage.measure import label

@sk.algorithm(tileable=True)
def label_objects(image, threshold=128):
    labels = label(image > threshold)
    return sk.Mask(labels, merger="instances")
```

See [Tile merging](../concepts/coordinates.md#tile-merging) for how each layer type is assembled from tiles.
