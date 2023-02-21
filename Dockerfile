FROM nvidia/cuda:11.6.0-devel-ubuntu18.04

COPY . .

RUN apt-get update
# add python repo if needed for correct version
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y

# install requirements
# noninteractive and TZ for tzdata install
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
RUN xargs apt-get install -y <packages.txt

RUN pip3 install -r requirements.txt
RUN pip3 install deep-rts/

RUN python3.10 src/main.py
