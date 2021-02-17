FROM ubuntu:20.04

# update aptitude
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich
RUN apt-get update && apt-get -y --fix-missing upgrade

# install aptitude essentials
RUN apt-get -y install \
    build-essential \
    meson \
    cmake \
    git \
    vim \
    curl \
    openmpi-bin \
    openmpi-common \
    libhdf5-openmpi-dev \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-matplotlib \
    python3-pandas \
    python3-scipy \
    python3-xlrd \
    python3-ipython \
    dirmngr apt-transport-https lsb-release ca-certificates
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get -y install nodejs

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install jupyter jupyterlab ipympl
RUN python3 -m pip install pylj
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install jupyter-matplotlib

ENTRYPOINT ["jupyter", "lab"]
CMD ["--ip=0.0.0.0", "--port=8888", "--no-browser"]

# run the container with:
# $ docker run -it --rm -p 8888:8888 <image name>

# add non-root user
RUN useradd -m pylj
WORKDIR /home/pylj
USER pylj
