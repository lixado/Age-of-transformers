# syntax=docker/dockerfile:1

FROM ubuntu:22.04

WORKDIR /age-of-transformers

RUN apt-get update && apt-get install -y curl wget ca-certificates zip python3-pip git ccache libgtk-3-dev bison cmake

COPY requirements.txt requirements.txt

COPY . .