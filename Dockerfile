FROM nvidia/cuda:11.6.0-devel-ubuntu18.04
# should problably change to pytorch
COPY . .

RUN apt-get update
# add python repo if needed for correct version
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y

# add cmake repo for newest version
RUN apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget -y
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' -y
RUN apt-get update


# install requirements
# noninteractive and TZ for tzdata install
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
RUN xargs apt-get install -y <packages.txt

RUN curl -O https://bootstrap.pypa.io/get-pip.py
RUN python3.10 get-pip.py

RUN pip install -r requirements.txt
RUN pip install deep-rts/

RUN python3.10 src/main.py
