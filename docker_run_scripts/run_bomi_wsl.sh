#!/bin/bash

# This script launches the container with the necessary flags from the WSL inside a Windows machine

DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0

MOUNT_WSLG=""

if [[ -d  /mnt/wslg ]]; then
	MOUNT_WSLG="-v /mnt/wslg/:/mnt/wslg/"
else
	echo "Running docker without wslg, you need an external xServer on Windows in order to see GUI windows."
fi


dpkg -l | grep nvidia-container-toolkit &> /dev/null
HAS_NVIDIA_TOOLKIT=$?
which nvidia-docker > /dev/null
HAS_NVIDIA_DOCKER=$?
if [ $HAS_NVIDIA_TOOLKIT -eq 0 ]; then
  docker_version=`docker version --format '{{.Client.Version}}' | cut -d. -f1`
  if [ $docker_version -ge 19 ]; then
	  DOCKER_COMMAND="docker run --gpus all"
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
 -dit --rm \
 --privileged \
 --env DISPLAY=$DISPLAY \
 --env "PULSE_SERVER=${PULSE_SERVER}" \
 ${MOUNT_WSLG} \
-p 8081:4242 \
 hypothe/bomi_fama