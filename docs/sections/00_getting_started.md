# Getting started

The following section will give you a first introduction to the Imaging Server Kit using a set of **interactive demos**.

## Prerequisites

Make sure to have installed the required packages:

```
pip install imaging-server-kit napari-serverkit
```

## Run the Napari demo

The Napari demo will give you a first idea of what the package can do. From a terminal, run:

```
serverkit demo napari
```

This command should open a **Napari viewer** with the *Server Kit* plugin already loaded. In the *Algorithm* dropdown, you will see a list of available algorithms.

![Demo-Napari](../assets/demo_napari_screenshot.png)

When you select an **algorithm**, the *Parameters* panel will update to display a list of tunable **parameters** for the algorithm.

Most algorithms require an input image. You can **load a sample image** from the *Samples* dropdown. Once you have loaded an image, you can **run** the algorithm and visualize the results in the Napari viewer.

![Napari-threshold](../assets/screenshot_napari_threshold.png)

Some algorithms automatically re-run when you change a parameters; for example, *Intensity threshold* updates the output directly when you adjust the threshold value.

```{admonition} Algo docs
You can access a **documentation page** for an algorithm in a web browser by clickin the **🌐 Doc** button. The documentation page provides a description of the algorithm as well as detailed information about its parameters. 
```

## Run the server demo

To explore how algorithms can be served over HTTP, start the local demo server:

```
serverkit demo serve
```

This launches a web server on your local machine at http://localhost:8000. If you open that page in your browser,you will see an overview of the algorithms available on the server.

![Server-page](../assets/screenshot_server.png)

### Connect from Napari

While your server is running, you can connect to it directly from Napari. Open another terminal and run:

```
napari -w napari-serverkit
```

This is equivalent to opening the plugin in Napari from `Plugins > Server Kit (Napari Server Kit)`.

In the plugin panel, enter the server address (http://localhost:8000) and press *Connect*. The *Algorithm* dropdown will populate with the available algorithms. You can use them just like in the local case.

<video width=640 controls loop autoplay>
  <source src="../_static/server_napari.mp4" type="video/mp4">
</video>

```{note}
In this demo, both client and server run on your local machine, but keep in mind that the server could also be hosted on another machine in your local network (workstation, cluster node, raspberry pi, etc.).
```

### Connect from QuPath

You can also use Server Kit algorithms direction from QuPath, for segmentation and object detection tasks. To try this out, follow the instructions in the [qupath-extension-serverkit](https://github.com/Imaging-Server-Kit/qupath-extension-serverkit) repository.

![threshold-qupath](../assets/threshold_qupath.png)

## Next steps

In the next section, you will learn to **create your own algorithm** in Python, so that it can be served, documented, and used in Napari just like the examples from the demo.