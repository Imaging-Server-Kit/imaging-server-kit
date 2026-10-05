# Data layers

## Data layers

A **data layer** holds a single piece of data, such as an image, a segmentation mask, a set of points, or a numeric value, together with information about what that data means. All data layers derive from `sk.Layer`, and the available types are listed in [Data layers](../reference/api/layers.md).

Data layers play two roles:

- **Describing parameters.** Layers passed in `parameters={}` describe the algorithm's inputs: their type, constraints such as `min` and `max`, a default value, a name, and a description. User interfaces and documentation pages are generated from these descriptions, and parameter values are validated against them.
- **Holding results.** Layers returned by an algorithm wrap its outputs, so that each output is displayed and handled correctly. For example, an array wrapped in `sk.Mask` is shown as a `Labels` layer in Napari.

Every layer has the following attributes:

| Attribute | Description |
|---|---|
| `data` | The data itself, for example a NumPy array or a number. |
| `name` | The name of the layer. Defaults to the name of the layer type, such as `"Mask"`. |
| `meta` | A dictionary of metadata: the description, the merging strategy, and any extra keyword arguments passed to the layer (for example `colormap="viridis"`). |
| `kind` | A short string identifying the layer type, such as `"mask"`. |

Spatial layers also have a position and an extent in global pixel coordinates (see [Coordinates, domains and tile merging](coordinates.md)).

## Stacks

A **stack** (`sk.Stack`) is an ordered collection of data layers. `.run()` returns the results of an algorithm as a stack, and `.get_sample()` returns samples as a stack of parameter layers.

```python
import imaging_server_kit as sk
import numpy as np

stack = sk.Stack([sk.Image(np.zeros((10, 10))), sk.Mask(np.ones((10, 10), dtype=int))])

image_layer = stack[0]           # By index
mask_layer = stack.read("Mask")  # By name
```

Within a stack, layer names are unique. When a layer is added with `stack.add()` under a name that is already taken, a suffix is appended to its name, for example `Image-01`.

When results are merged into a stack, layers are matched by name:

- A layer with a *new name* is **added** to the stack.
- A layer with the *same name* as an existing layer **updates** that layer's data and metadata.

This concept is the basis of [live updates](../how-to/live-updates.md) and [tiled inference](../how-to/tiling.md); each yielded output, or processed tile, is merged into a single result stack according to the rule above, based on layer name.