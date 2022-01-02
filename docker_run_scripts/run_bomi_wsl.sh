#!/bin/bash

# This script launches the container with the necessary flags from the WSL inside a Windows machine

DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0

docker run -dit --rm --gpus all --privileged --env NVIDIA_DRIVER_CAPABILITIES='graphics,compute,utility' \
 --env DISPLAY=$DISPLAY \
 --env "PULSE_SERVER=${PULSE_SERVER}" -v /mnt/wslg/:/mnt/wslg/ \
 hypothe/bomi_fama:latest