FROM python:3.9.6

RUN apt-get update
# Set the working directory to /app
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0"]
