![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# 🪐 Imaging Server Kit

Turn Python-based image processing workflows into **algorithms** that gain extra functionalities.

- [**Turn your algorithms into web servers**](https://imaging-server-kit.github.io/imaging-server-kit/sections/07_server.html) and run computations from [QuPath](https://github.com/Imaging-Server-Kit/qupath-extension-serverkit), [Napari](https://github.com/Imaging-Server-Kit/napari-serverkit), or [Python](./sections/08_python) via HTTP requests.

https://github.com/user-attachments/assets/36bda69d-996e-4240-9a53-7e76d9ea3894

- [**Generate dock widgets**](https://imaging-server-kit.github.io/imaging-server-kit/sections/01_algorithm.html) to run your algorithms interactively in Napari.

https://github.com/user-attachments/assets/1ff572f7-f159-4f5a-afd4-7a157de3d9f8

- Run your algorithms [**tile-by-tile**](https://imaging-server-kit.github.io/imaging-server-kit/sections/06_tiled.html) on the input data.

https://github.com/user-attachments/assets/47c2f734-5683-49d9-8aea-388c3a2bc16d

- [**Stream results**](https://imaging-server-kit.github.io/imaging-server-kit/sections/05_streams.html) to inspect them in real-time.

https://github.com/user-attachments/assets/a3f69a9f-fb68-4580-a804-6c57d5807b9a

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

## Getting started

The documentation is available on [this page](https://imaging-server-kit.github.io/imaging-server-kit).

## Contributing

Contributions are very welcome.

## License

This software is distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license.

## Issues

If you encounter any problems, please file an issue along with a detailed description.

## Citing

[![DOI](https://zenodo.org/badge/912741131.svg)](https://doi.org/10.5281/zenodo.15673151)

If you use the Imaging Server Kit in the context of scientific publication, you can cite it as below.

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
