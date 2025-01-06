# Doc: https://github.com/conda-forge/miniforge-images
FROM condaforge/miniforge3:latest

ARG PYTHON_VERSION=3.9
RUN conda install python=$PYTHON_VERSION -y

WORKDIR /app

COPY . .

RUN python -m pip install -e .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
