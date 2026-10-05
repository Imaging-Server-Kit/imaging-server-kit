# Restrict computation to a region

Large images can be processed only in a **region of interest**. Regions are described with `sk.Domain` objects, which represent a sub-region of global pixel space (see [Coordinates and domains](../concepts/coordinates.md)).

## Defining a region

A domain is a box defined by a `position` (its lower corner, for example the top-left corner in 2D) and a `size`, in global pixel coordinates:

```python
import imaging_server_kit as sk

roi = sk.Domain(position=(20, 30), size=(60, 80))

print(roi.coords_min)  # (20.0, 30.0)
print(roi.coords_max)  # (80.0, 110.0)
```

## Restricting a computation

You can pass the domain to `.run()` with `domain=`. The algorithm only runs on the inputs inside that region, and the results are placed at the position of the domain:

```python
from imaging_server_kit.demo.tools import otsu_threshold
from skimage.data import coins

# Run Otsu's threshold on a sub-region of the image
results = otsu_threshold.run(image=coins(), domain=roi)

print(results[0])  # <Mask 'Binary mask (auto)' bool (60, 80), 1 labels, at (20, 30)>
```

`domain=` can be combined with [tiling](tiling.md) for tileable algorithms, and works with clients connected to a server too.

## Selecting part of a layer or stack

You can extract the part of a layer, or of a stack, that falls inside a domain with `.select()`:

```python
image = sk.Image(coins())

image_roi = image.select(roi)

print(image_roi)  # <Image 'Image' uint8 (60, 80), at (20, 30)>
```

The selection will keep its position in global coordinates.

## Indexing in layers and stacks

Layers and stacks also support NumPy-like indexing. With stacks, the first index selects layers (next ones are spatial):

```python
import numpy as np

img = sk.Image(np.zeros((200, 200)))
pts = sk.Points(np.array([[30, 10], [60, 90], [120, 110]]))

print(img[50:150])  # <Image 'Image' float64 (100, 200), at (50, 0)>
print(pts[50:150])  # <Points 'Points' 2 points, 2D, at (50, 0)>

stack = sk.Stack([img, pts])

roi_layers = stack[:, 50:150]  # <- All layers, rows 50 to 150

print(roi_layers)  # [<Image 'Image' float64 (100, 200), at (50, 0)>, <Points 'Points' 2 points, 2D, at (50, 0)>]
```
