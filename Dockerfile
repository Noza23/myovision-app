FROM python:3.9.6

RUN apt-get update

WORKDIR /app

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /app

EXPOSE 8080

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0" , "--port", "8080"]
