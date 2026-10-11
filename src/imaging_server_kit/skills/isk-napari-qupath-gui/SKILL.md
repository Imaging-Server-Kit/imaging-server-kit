---
name: isk-napari-qupath-gui
description: Build and run interactive Napari or QuPath interfaces for Imaging Server Kit algorithms with sk.to_napari, sk.to_qwidget, sk.to_qupath, the `serverkit` CLI and the Napari plugin menu, including auto-updating parameters, tiled inference widgets, showing results as layers, and connecting to a remote algorithm server. Use this whenever the user wants to try, inspect, visualize or tune an imaging_server_kit / `@sk.algorithm` algorithm interactively, wants a Napari dock widget or QuPath panel for it, or asks how results are displayed in Napari or QuPath.
---

# Napari and QuPath interfaces

Interactive inspection matters in image analysis: the goal is to let the user move a slider and see the mask change. The widgets are generated from the algorithm's annotations, so most GUI problems are annotation problems (see the `isk-write-algorithm` skill).

## Napari

Install: `pip install "imaging-server-kit[napari]"`.

```python
import imaging_server_kit as sk
import skimage.data

viewer = sk.to_napari(my_algo)          # new viewer with the widget docked
viewer.add_image(skimage.data.coins())  # something to run on
```

- Existing viewer: `sk.to_napari(my_algo, viewer=viewer)`.
- Accepts a single algorithm, a collection (`sk.combine`), or a client (`sk.Client(url)`); the widget behaves the same.
- In a plain script, end with `napari.run()` or the window closes immediately (not needed in Jupyter/IPython).
- Embedding in another Napari plugin: `sk.to_qwidget(algo, viewer)` returns the widget without docking it.
- Command line: `napari -w imaging-server-kit` opens the "connect to server" widget; `serverkit demo napari` and `serverkit tools napari` open built-in algorithm sets.
- Run from Python and push into a viewer: `algo.run(image, stack=viewer)`; `sk.convert(stack, to="napari")` opens a stack in a new viewer.

### How annotations become widgets

| Parameter layer | Field |
|---|---|
| `sk.Integer`, `sk.Float` | spin box (`min`, `max`, `step`, `default`) |
| `sk.Bool` | checkbox |
| `sk.Choice` | dropdown of `items` |
| `sk.String` | text field |
| `sk.Image`, `sk.Mask`, `sk.Points`, `sk.Boxes`, `sk.Vectors`, `sk.Paths`, `sk.Tracks` | dropdown of matching viewer layers |
| `sk.Any`, `sk.Null` | hidden |

`name` is the label and `description` the tooltip. `auto_call=True` re-runs the algorithm when the value changes. Samples appear in a *Samples* dropdown with a Load button; `tileable=True` adds a *Tiled inference* section.

### How outputs are displayed

`sk.Image` is an Image layer, `sk.Mask` a Labels layer, `sk.Points` Points, `sk.Boxes`/`sk.Paths` Shapes, `sk.Vectors` Vectors, `sk.Tracks` Tracks, simple values a text overlay, `sk.Notification` a Napari notification, `sk.Progress` the progress bar. Re-running updates layers with the same name instead of adding new ones, so keep output names stable. Extra kwargs such as `colormap="viridis"` become layer properties.

### Troubleshooting

- Segmentation shows as grayscale: the output was returned as a bare array; wrap it in `sk.Mask`.
- Widget shows no field for a parameter: it resolved to `sk.Any`; annotate it.
- Slider re-runs a slow model on every tick: drop `auto_call=True` for that parameter.
- Nothing happens on Run: check that the input dropdown has a layer of the right kind selected in the viewer.
- Napari not installed: the error says to install the `napari` extra.

## QuPath

QuPath support is experimental and narrower than Napari. Tell the user this rather than promising full coverage.

Requirements: `pip install "imaging-server-kit[qupath]"` (Python 3.11+), and the `qupath-extension-py4j` extension in QuPath with a Py4J gateway started.

- The algorithm must take **exactly one `sk.Image` input**; it receives the full-resolution image inside the selected rectangular annotation.
- Displayable outputs: `sk.Mask` (converted to polygon annotations) and `sk.Boxes`. Anything else needs a Napari viewer alongside: `serverkit qupath --with-napari`, or `viewer=` in `sk.to_qupath`.
- Local algorithm: `sk.to_qupath(algo)` inside an `if __name__ == "__main__":` block; options `port=` and `token=` must match the Py4J gateway.
- Server algorithms: start a server (`serverkit demo serve` or `sk.serve(algo)`), run `serverkit qupath`, click *Connect to QuPath* (pick the annotation class), then *Connect* to the server and run.
- Tileable algorithms (`tileable=True`) are a good fit, since QuPath images can be very large.
