FROM python:3.9.6


RUN apt-get update
# Set the working directory to /app
WORKDIR /app
COPY . /app
RUN pip install 'git+https://${TOKEN}@github.com/Noza23/myovision-sam.git@9-inference-pipeline'
RUN pip install fastapi[all]

CMD ["uvicorn", "backend.main:app", "--reload", "--port", "8000"]