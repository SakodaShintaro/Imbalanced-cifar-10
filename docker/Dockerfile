FROM nvidia/cuda:10.2-cudnn7-devel

RUN apt-get update && apt-get install -y \
    sudo \
    git \
    vim \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    torch==1.7.1 \
    torchvision \
    pandas \
    matplotlib \
    requests

WORKDIR /root