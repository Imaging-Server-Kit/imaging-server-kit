# Doc: https://github.com/conda-forge/miniforge-images
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN python3 -m pip install --upgrade pip
RUN pip install imaging-server-kit cellpose opencv-python-headless

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["python3", "main.py"]