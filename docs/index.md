# Welcome to the Imaging Server Kit's documentation!

The **Imaging Server Kit** lets you turn Python-based image processing workflows into **algorithms** that gain extra functionalities.

```python
@sk.algorithm  # <- Turn your function into an algorithm
def my_algo(image, parameter):
  (...)
```

Server Kit algorithms are versatile objects that allow you run computations in a variety of ways.

For example, you can

- [**Turn your algorithm into a web server**](./sections/07_server.md), connect to it and run computations from [QuPath](https://github.com/Imaging-Server-Kit/qupath-extension-serverkit), [Napari](https://github.com/Imaging-Server-Kit/napari-serverkit), or [Python](./sections/08_python) via HTTP requests.

<video width=512 controls loop autoplay>
  <source src="./_static/stardist_blobs.mp4" type="video/mp4">
</video>

- [**Generate a dock widget**](./sections/01_algorithm) to run your algorithm interactively in Napari.

<video width=512 controls loop autoplay>
  <source src="./_static/oripy_threshold.mp4" type="video/mp4">
</video>

- Run your algorithm [**tile-by-tile**](./sections/06_tiled) on the input image.

<video width=512 controls loop autoplay>
  <source src="./_static/tiles.mp4" type="video/mp4">
</video>

- [**Stream results**](./sections/05_streams) to inspect them in real-time.

<video width=512 controls loop autoplay>
  <source src="./_static/yolo-stream.mp4" type="video/webm">
</video>

On top of that, you can [**validate data**](./sections/08_python) passed as algorithm parameters, provide [**samples**](./sections/02_samples), and generate [**documentation**](./sections/03_metadata) for your algorithm that you can share with users.

This documentation will give you a conceptual overview of the package, and walk you through the steps to learn [how to create an algorithm](./sections/01_algorithm), and give you some [suggestions of use cases](./sections/11_examples).

## Contents

```{tableofcontents}
```

## Installation

Install the `imaging-server-kit` package with `pip`:

```
pip install imaging-server-kit
```

or clone the project and install the development version:

```
git clone https://github.com/Imaging-Server-Kit/imaging-server-kit.git
cd imaging-server-kit
pip install -e .
```

**Napari plugin**

Install the [napari-serverkit](https://github.com/Imaging-Server-Kit/napari-serverkit) plugin with `pip` to enable all [Napari](https://napari.org/stable/)-related functionalities.

```sh
pip install napari-serverkit
```

**QuPath extention**

To use the Imaging Server Kit in [QuPath](https://qupath.github.io/), follow the installation instructions at [qupath-extension-serverkit](https://github.com/Imaging-Server-Kit/qupath-extension-serverkit).

## License

This software is distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license.

## Citing

If you use `imaging-server-kit` in the context of scientific publication, you can cite it as below.

BibTeX:

```
@software{mallory_wittwer_2025_15673152,
  author       = {Mallory Wittwer and Edward Andò and Maud Barthélemy and Florian Aymanns},
  title        = {Imaging-Server-Kit/imaging-server-kit: v0.0.14},
  url          = {https://doi.org/10.5281/zenodo.15673152},
  doi          = {10.5281/zenodo.15673152},
  version      = {v0.0.14},
  year         = 2025,
}
```

## Acknowledgements

We thank the [Personalized Health and Related Technologies](https://www.sfa-phrt.ch/) for funding this project.