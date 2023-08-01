FROM nvcr.io/nvidia/pytorch:23.07-py3

COPY requirements-docker.txt ./requirements-docker.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements-docker.txt