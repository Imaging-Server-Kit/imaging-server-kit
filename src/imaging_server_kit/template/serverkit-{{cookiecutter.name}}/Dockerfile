# Doc: https://github.com/conda-forge/miniforge-images
FROM condaforge/miniforge3:latest

ARG PYTHON_VERSION={{ cookiecutter.python_version }}
RUN conda install python=$PYTHON_VERSION -y

WORKDIR /app

COPY . .

RUN python -m pip install -r requirements.txt

EXPOSE 8000

CMD ["python3", "main.py"]