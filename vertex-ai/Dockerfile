FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-gpu.2-15.py310
#FROM python:3.8

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src .
ENV PYTHONPATH=/


