# Stream live updates

Algorithms can send **intermediate outputs** before they finish running. This lets user interfaces such as Napari update in real time.

For example, you might stream:

- Frames from a camera, with a segmentation overlay or detection boxes.
- Progress notifications from a long-running task.
- Intermediate results of an iterative algorithm.

## Yielding outputs

When a function decorated with `@sk.algorithm` contains `yield` statements, the yielded values are treated as **intermediate outputs**. The yielded values should be data layers, just like return values.

Layers with the same name are **updated** (rather than added), so only the last value of each layer remains in the final [stack](../concepts/layers-and-stacks.md).

Here is an algorithm that progressively increases a threshold:

```python
import time

import imaging_server_kit as sk
import numpy as np
import skimage.data

@sk.algorithm
def threshold_algo(image, steps=20):
    for k, threshold in enumerate(np.linspace(0, 255, steps)):  # Progressively increase the threshold
        mask = image > threshold
        yield sk.Mask(mask), sk.String(f"Threshold: {threshold}")  # <- Stream the outputs
        yield sk.Progress(k, max_val=steps)  # <- Update the progress bar
        time.sleep(0.5)

viewer = sk.to_napari(threshold_algo)
viewer.add_image(skimage.data.coins())
```

In Napari, the segmentation mask is updated as the threshold increases, and the progress bar of the widget fills up progressively.

<video width="512" controls loop autoplay muted playsinline>
  <source src="../assets/videos/progressive_threshold.mp4" type="video/mp4">
</video>

## Combining `yield` and `return`

An algorithm can yield intermediate outputs and still produce a final result with `return`:

```python
import time

import imaging_server_kit as sk
import skimage.data

@sk.algorithm
def mini_pipeline(image, threshold=100):
    yield sk.Notification("Starting the processing")

    time.sleep(2)

    mask = image > threshold

    yield sk.Notification("Segmented the image"), sk.Mask(mask)

    time.sleep(2)

    fract = mask.sum() / mask.size

    return sk.Notification(f"Fraction of True pixels: {fract}")

viewer = sk.to_napari(mini_pipeline)
viewer.add_image(skimage.data.coins())
```

Here, the algorithm notifies the user that processing has started, then yields a segmentation mask, and finally reports a measurement.

## Notifications and progress

- `sk.Notification` is used to show a message to the user. In Napari, it appears as a notification. You can set its level with `level="info"` (default), `"warning"`, or `"error"`.
- `sk.Progress` updates a progress bar, up to `max_val`.

See [Data layers](../reference/api/layers.md) for the other layer types.
