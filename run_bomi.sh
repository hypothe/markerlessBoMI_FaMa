xhost +local:docker
docker run \
    --privileged -it --rm --gpus all \
    --env NVIDIA_DRIVER_CAPABILITIES='graphics,compute,utility' \
    --env DISPLAY=$DISPLAY --env XDG_RUNTIME_DIR --env QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix --network=host  \
    hypothe/bomi_fama
