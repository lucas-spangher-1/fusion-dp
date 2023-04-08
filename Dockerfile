FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.7.1

# Working directory
WORKDIR .

# Install dependencies
RUN pip install -r requirements-docker.txt

