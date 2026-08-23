#!/bin/bash

# Accept arguments
username="${DOCKER_USERNAME:-$(whoami)}"

# CUDA architecture settings
# You can find the list of CUDA architectures here: https://developer.nvidia.com/cuda-gpus
arch="${1:-8.6+PTX}"

# Construct image name
image_name="$username/phystwin:1.0"

# Print build information
echo "Building Docker image with:"
echo "  Username: $username"
echo "  CUDA Architecture: $arch"
echo "  Image Name: $image_name"

# Build the Docker image
docker build --build-arg TORCH_CUDA_ARCH_LIST=$arch -t $image_name . || { echo "Docker build failed"; exit 1; }