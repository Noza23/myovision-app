
FROM python:3.9.6

RUN apt-get update
# Set the working directory to /app
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0"]
