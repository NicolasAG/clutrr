FROM python:3.7
COPY . /clutrr/
RUN cd /clutrr && pip install -r requirements.txt
RUN cd /clutrr && python setup.py develop
