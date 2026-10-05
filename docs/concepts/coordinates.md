# Pixel space

## Coordinates

Spatial layers (images, masks, points, etc.) have a `position`, which places them in **global pixel coordinates**. A layer's data is expressed in *local* coordinates, and `position` is the offset that translates them to global coordinates.

For example, an image cropped from a larger image keeps track of where the crop came from: its data starts at index `(0, 0)`, while its `position` records where that pixel is in the original image.

You can set the position when creating a layer:

```python
import imaging_server_kit as sk
import numpy as np

img = sk.Image(np.zeros((100, 200)), position=(50, 20))

print(img)             # <Image 'Image' float64 (100, 200), at (50, 20)>
print(img.position)    # (50, 20)
print(img.extent)      # Domain(position=(50, 20), size=(100, 200))
print(img.coords_min)  # (50.0, 20.0)
print(img.coords_max)  # (150.0, 220.0)
```

For object layers such as points, the extent is the bounding box of the objects:

```python
pts = sk.Points(np.array([[30, 10], [60, 90]]))

print(pts.position)    # (0, 0)
print(pts.coords_min)  # (30.0, 10.0)
print(pts.coords_max)  # (61.0, 91.0)
```

Stacks have a position and an extent too. Setting the position of a stack moves all of its layers:

```python
stack = sk.Stack([img, pts])

print(stack.extent)  # Domain(position=(30, 10), size=(120, 210))

stack.position = (10, 10)

print(img.position)  # (60, 30)
print(pts.position)  # (10, 10)
```

Layers and stacks share the following properties:

| Property | Type | Description |
|---|---|---|
| `position` | `tuple` | Offset of the local coordinates in global pixel coordinates. |
| `extent` | `sk.Domain` | Region covered by the data, in global coordinates. |
| `coords_min` | `tuple` | Lower corner of the extent (top-left in 2D). |
| `coords_max` | `tuple` | Upper corner of the extent (bottom-right in 2D). |
| `size` | `tuple` | Size of the extent along each axis. |
| `ndim` | `int` | Number of spatial dimensions. |

## Domains

Extents are `sk.Domain` objects. A domain is a box defined by a `size` and a `position` (its lower corner, which defaults to the origin):

```python
roi = sk.Domain(position=(20, 30), size=(60, 80))

print(roi.coords_min)  # (20.0, 30.0)
print(roi.coords_max)  # (80.0, 110.0)
```

Domains are used to describe regions of interest and to [restrict computations](../how-to/regions.md), for example.

## Merging data

### Merging tiles

When an algorithm runs [tile-by-tile](../how-to/tiling.md), the input layers are split into tiles, the algorithm runs on each tile, and the results are merged into a single stack. How results are merged depends on the layer type:

| Layer | Merging |
|---|---|
| `sk.Image` | Intensities are **averaged** in overlapping regions. |
| `sk.Mask` | By default, the **last tile overwrites** overlapping regions. With `merger="instances"` and a non-zero overlap between tiles, labels are made unique across tiles and objects crossing tiles are stitched together (experimental). |
| `sk.Points`, `sk.Vectors`, `sk.Boxes` | Each object belongs to the tile that contains it (vectors are placed based on their origin, boxes based on their center). When tiles overlap, objects in overlapping regions are returned once per tile. |
| Other layers (`sk.Paths`, `sk.Tracks`, values, etc.) | Each tile **replaces** the previous value. |

### Merging layers

You can use `sk.merge_layers()` to merge a list of layers of the same kind into a new layer, using the rules above. For example, to assemble two images:

```python
left = sk.Image(np.ones((100, 100)))
right = sk.Image(np.full((100, 100), 2.0), position=(0, 100))

merged = sk.merge_layers([left, right])

print(merged)           # <Image 'Image' float32 (100, 200)>
print(merged.position)  # (0.0, 0.0)
```
