FROM ubuntu:22.04

COPY . .

RUN apt-get update
RUN xargs apt-get install -y <packages.txt

RUN pip install -r requirements.txt
RUN pip install deep-rts/

RUN python3 src/main.py
