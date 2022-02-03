#!/bin/bash

# This script launches the container with the necessary flags on a Unix machine

xhost +local:docker

NET_NAME="bomi-fama"

docker network inspect ${NET_NAME} --format {{.Id}} 2>/dev/null || docker network create --driver bridge ${NET_NAME}

DOCKER_ENV="--env DISPLAY=$DISPLAY --env XDG_RUNTIME_DIR --env QT_X11_NO_MITSHM=1"


dpkg -l | grep nvidia-container-toolkit &> /dev/null
HAS_NVIDIA_TOOLKIT=$?
which nvidia-docker > /dev/null
HAS_NVIDIA_DOCKER=$?
if [ $HAS_NVIDIA_TOOLKIT -eq 0 ]; then
  docker_version=`docker version --format '{{.Client.Version}}' | cut -d. -f1`
  if [ $docker_version -ge 19 ]; then
	  DOCKER_COMMAND="docker run --gpus all"
    DOCKER_ENV+=" --env NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility"
  else
	  DOCKER_COMMAND="docker run --runtime=nvidia"
  fi
elif [ $HAS_NVIDIA_DOCKER -eq 0 ]; then
  DOCKER_COMMAND="nvidia-docker run"
else
  echo "Running without nvidia-docker, if you have an NVidia card you may need it"\
  "to have GPU acceleration"
  DOCKER_COMMAND="docker run"
fi

$DOCKER_COMMAND \
  --privileged -it --rm \
  ${DOCKER_ENV} \
  --network=${NET_NAME} \
  --env BOMI_SERVER_NAME=bomi_server \
  -v /tmp/.X11-unix:/tmp/.X11-unix --network=host  \
  hypothe/bomi_fama
