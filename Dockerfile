FROM python:3.7
COPY . /data/clutrr/
RUN cd /data/clutrr && pip install -r requirements.txt
RUN cd /data/clutrr && python setup.py develop
