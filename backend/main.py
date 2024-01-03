from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application")
    # Load models, connect to database, etc.

    yield
    # Clean up
    print("Stopping application")


app = FastAPI(lifespan=lifespan)


@app.get("/run")
async def run():
    # Run the pipeline
    return {"message": "Hello World"}
