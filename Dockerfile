FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY requirements-docker.txt ./requirements-docker.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements-docker.txt