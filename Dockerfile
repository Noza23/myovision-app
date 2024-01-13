FROM python:3.9.6

# Set the working directory to /app
WORKDIR /app
COPY . /app
RUN pip install .

CMD ["uvicorn", "backend.main:app", "--reload", "--port", "8000"]