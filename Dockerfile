# Start with the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    freeglut3-dev \
    libglib2.0-0 \
    libxcb-util1 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxkbcommon-x11-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Qt
ENV QT_DEBUG_PLUGINS=1
ENV QT_QPA_PLATFORM=xcb

# Install miniconda
ENV CONDA_DIR="/opt/conda"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash 

# Create a new conda environment
RUN /opt/conda/bin/conda create -y -n phystwin_env python=3.10

# Set the working directory, non-root user, and permissions
WORKDIR /PhysTwin

# Copy contents of the repository to the container
COPY --chmod=755 . .

# CUDA architecture settings
# This is set to 8.6 for NVIDIA RTX 30 series GPUs (Ampere architecture)
# If you are using a different GPU, make sure to set this to the correct architecture
# You can find the list of CUDA architectures here: https://developer.nvidia.com/cuda-gpus
ARG TORCH_CUDA_ARCH_LIST="8.6+PTX"

# Activate the conda environment and install dependencies
RUN /bin/bash -c "source activate phystwin_env && chmod +x env_install/env_install.sh && ./env_install/env_install.sh"

# Set the default command
CMD ["/bin/bash"]