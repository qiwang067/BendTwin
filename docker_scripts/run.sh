#!/bin/bash

# Load arguments
username="${DOCKER_USERNAME:-$(whoami)}"
data_dir=$1
experiments_dir=$2
experiments_optimization_dir=$3
gaussian_output_dir=$4

# Check if the Docker image is available
if ! docker image inspect $username/phystwin:1.0 > /dev/null 2>&1; then
    echo "Docker image $username/phystwin:1.0 not found. Please build the image first."
    exit 1
fi

# Check if the X11 server is running
if ! pgrep -x "Xorg" > /dev/null; then
    echo "X11 server is not running. Please start the X11 server first."
    exit 1
fi

# Allow access to the X11 server for local connections
# This command allows the root user to access the X11 server
# without needing to enter a password. This is necessary for GUI applications
# running inside the Docker container to display on the host machine.
# Note: This command may have security implications, so use it with caution.
# It is recommended to use this command only in a trusted environment.
xhost +local:root

docker run --gpus 'all,"capabilities=compute,utility,graphics"' \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
    -v $data_dir:/PhysTwin/data \
    -v $experiments_dir:/PhysTwin/experiments \
    -v $experiments_optimization_dir:/PhysTwin/experiments_optimization \
    -v $gaussian_output_dir:/PhysTwin/gaussian_output \
    --privileged --entrypoint /bin/bash -it $username/phystwin:1.0