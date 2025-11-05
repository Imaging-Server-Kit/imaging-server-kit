# Using Docker

Serving algorithms in Docker containers can simplify their usage and deployment on different machines.

## Start from the template

To help you create a standard layout for your Imaging Server Kit project, you can use our **cookiecutter template**.

The template includes:

- A `README.md` file to describe your project.
- A `requirements.txt` file to list dependencies. 
- A recommended `.gitignore` file.
- Docker files: `Dockerfile`, `docker-compose.yml`, and `.dockerignore` to build and run your algorithm server.
- A `main.py` file where to implement your algorithm.
- A `samples/` folder where to include sample images.

Create a new project by running:

```
serverkit new <output_directory>
```

This generates a structure like:

```
serverkit-project
├── sample_images               # Sample images
│   └── blobs.tif
├── .gitignore
├── docker-compose.yml          # Run with `docker compose up`
├── Dockerfile
├── main.py                     # Algorithm implementation
├── README.md
└── requirements.txt
```

You will be prompted for a project name, project URL, and Python version. After generating your project, implement your algorithm in `main.py`. Consider adjusting `requirements.txt`, `Dockerfile`, `README.md` and other files as needed.

## Build and run with docker-compose

To build a docker image image and run your algorithm as a web server in a docker container, use the command:

```sh
docker compose up
```

This will build the docker image, install your algorithm, start the server and make it available on http://localhost:8000.

## Next steps

Congratulations! You have seen all of the main features of *Imaging Server Kit*.

The last two sections of this documentation explore:

- [Data layers](./10_types) - A reference of all available data layers.
- [Examples](./11_examples) - A gallery of example Server Kit algorithms for selected projects.