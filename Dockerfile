FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive


RUN apt update && apt install openslide-tools gcc build-essential libgl1-mesa-glx libssl-dev libbz2-dev python3 python3-pip -y
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /opt/app


# Copy the local package files to the container's workspace.
ADD . /app

# FROM  pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel


# Use ENTRYPOINT to allow arguments to be passed
ENTRYPOINT ["python"]