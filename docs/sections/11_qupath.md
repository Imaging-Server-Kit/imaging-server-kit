# Usage with QuPath

Imaging Server Kit provides a bridge to run algorithms on QuPath images via [QuBaLab](https://pypi.org/project/qubalab/). To use this functionality, install the optional QuPath dependencies via `pip install imaging-server-kit[qupath]`.

To connect to an algorith server *and* QuPath, run the command:

```sh
serverkit qupath
```

Alternatively, use `to_qupath` from Python (with an active Py4J gateway):

```python
import imaging_server_kit as sk

@sk.algorithm(tileable=True)
def threshold_manual(image, threshold: float = 0.5):
    thresh_rel = threshold * (image.max() - image.min())
    mask = image > thresh_rel
    return sk.Mask(mask)

if __name__ == "__main__":
    sk.to_qupath(threshold_manual)
```

Algorithms compatible with Qupath **must take exactly one `sk.Image` as input** (interpreted as the QuPath image). They are run on the full-resolution image inside a rectangular region of interest defined by a QuPath annotation. For example, you can create a rectangle in QuPath, assign it the class `Region`, find it in the dropdown list, and use it as a ROI for the computation.

