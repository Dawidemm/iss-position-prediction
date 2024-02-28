FROM python:3.10

WORKDIR /iss

COPY . /iss

RUN pip install --upgrade pip && \
    pip install .

EXPOSE 3000