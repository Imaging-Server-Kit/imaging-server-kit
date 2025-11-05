# Serving algorithms

Any Imaging Server Kit algorithm can be **served as a web API** using a built-in [FastAPI](https://fastapi.tiangolo.com/) server. This turns algorithms into web servers that you can interact with from Napari, QuPath, or Python via HTTP requests.

Serving an algorithm is particularly useful when:

- You want to run computations on a remote machine and visualize the results on your local machine.
- You want to use algorithms in **QuPath** via the `qupath-extension-serverkit` extension.

## Using `sk.serve`

Let's consider the threshold algorithm once again. You can serve this algorithm by passing it to `sk.serve()`. 

```python
import imaging_server_kit as sk
import skimage.data

@sk.algorithm(
    name="Intensity threshold",
    parameters={"threshold": sk.Integer(name="Threshold", min=0, max=255, default=128)},
    samples=[{"image" : skimage.data.coins()}],
)
def threshold_algo(image, threshold):
    mask = image > threshold
    return sk.Mask(mask, name="Binary mask")

if __name__ == "__main__":
    sk.serve(threshold_algo)  # <- Serve the algorithm
```

```{important}
`sk.serve()` won't work in a Jupyter notebook. You need to run the code as a **Python script**.
```

Calling `sk.serve()` starts a local FastAPI server exposing your algorithm via a set of predefined routes.

![Server-running](../assets/server_running.png)

By default, the server is hosted at http://localhost:8000. If you navigate to this URL in a browser, you will see the familiar **algorithm doc** page.

Once the server is running, you can connect to it from Napari, QuPath, or directly from Python.

### Connecting from Napari

The [Napari Server Kit](https://github.com/Imaging-Server-Kit/napari-serverkit) plugin provides a widget for connecting to algorithm servers.

To try it out, open a new terminal window, start Napari, and then open the plugin from `Plugins > Server Kit (Napari Server Kit)`. In the `Server URL` field, enter: http://localhost:8000, and press *Connect*. You should see your threshold algorithm listed, with all related functionalities (load samples, run algorithm, access documentation) available.

### Connecting from QuPath

You can also connect to algorithm servers via QuPath through the [qupath-extension-serverkit](https://github.com/Imaging-Server-Kit/qupath-extension-serverkit) extension. The QuPath extension can handle compatible algorithm outputs, including segmentation masks, bounding boxes, vectors, points, and classification tasks.

## Summary

- Use `sk.serve()` to expose any Imaging Server Kit algorithm as a FastAPI web service.
- The server runs locally at http://localhost:8000 by default.
- You can connect to algorithm servers from Napari or QuPath.

## Next steps

In the next section, we will explore how to interact with Server Kit algorithms directly from Python.