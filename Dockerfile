FROM python:3.9.6

RUN apt-get update
# Set the working directory to /app
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
# RUN pip install stardist
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# RUN pip install 'git+https://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@github.com/Noza23/myovision-sam.git@9-inference-pipeline'
# RUN pip install fastapi[all]

CMD ["uvicorn", "backend.main:app", "--port", "8000"]
