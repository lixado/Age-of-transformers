FROM python:latest

COPY . .

RUN apt-get update && apt-get install -y xargs

RUN xargs apt-get install -y <packages.txt

RUN pip install -r requirements.txt
RUN pip install deep-rts/