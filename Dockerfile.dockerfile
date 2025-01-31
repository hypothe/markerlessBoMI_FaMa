FROM tensorflow/tensorflow:latest-gpu

ENV HOME=/root
ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

ENV WS=${HOME}/biorob

WORKDIR ${HOME}

RUN	apt-get update && apt -y install python3-pip \
	&& apt install -y build-essential libsm6 libssl-dev libffi-dev python3-dev python3-tk ffmpeg xauth xxd

COPY . ${WS}

# necessary, somehow, for the xserver called by pyautogui
ENV XAUTHORITY=${HOME}/.Xauthority
RUN touch ~/.Xauthority \
	&& echo 'xauth generate :0 . trusted && \
			xauth add ${HOST}:0 . $(xxd -l 16 -p /dev/urandom)' >> ${HOME}/.bashrc

WORKDIR ${WS}
RUN pip install --upgrade pip \
	&& pip install -r requirements.txt

ENTRYPOINT [ "bash" ]