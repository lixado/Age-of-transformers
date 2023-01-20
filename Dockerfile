# syntax=docker/dockerfile:1

FROM ubuntu:22.04

WORKDIR /age-of-transformers

RUN apt-get update && apt-get install -y curl wget ca-certificates zip python3-pip git ccache libgtk-3-dev cmake bison

COPY requirements.txt requirements.txt

COPY . .

RUN pip3 install deep-rts/
