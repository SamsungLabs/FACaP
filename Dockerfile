FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update &&\
    apt-get upgrade -y &&\
    apt-get install -y curl git vim htop wget bzip2 g++ libgl1-mesa-dev libosmesa6-dev patchelf libglfw3 libglew-dev libglib2.0-0 libsm6 unzip xvfb

ARG USERNAME=user
RUN apt-get install -y sudo && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME

USER $USERNAME:$USERNAME
WORKDIR "/home/$USERNAME"
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

ENV PATH=/home/$USERNAME/miniconda3/bin:${PATH}

RUN wget -q "https://repo.continuum.io/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh" -O 'miniconda3.sh' && \
    bash 'miniconda3.sh' -b -p '/home/$USERNAME/miniconda3' && \
    rm 'miniconda3.sh' && \
    conda update -y conda && \
    conda install pip

ENV HOME /home/$USERNAME
ADD env.yaml env.yaml
RUN conda env update -f env.yaml --prune

RUN pip install open3d==0.10.0 -U


SHELL ["/bin/bash", "-c"]


    


    


