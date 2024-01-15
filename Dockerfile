FROM python:3.10.9

WORKDIR /iss

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY datasets datasets
COPY src src

WORKDIR /iss/src

CMD ["python", "main.py"]