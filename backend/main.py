from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import Settings, REDIS_KEYS, Config
import random

settings = Settings(_env_file=".env", _env_file_encoding="utf-8")
# pipeline: Pipeline = None
redis_keys = REDIS_KEYS(prefix="myovision")
origins = ["http://localhost:3000"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan of application."""
    print("Starting application")
    # Redis connection
    print("Connecting to redis")
    # redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    # Loading models
    print("Loading models")
    # global pipeline
    # pipeline = Pipeline()
    # pipeline._stardist_predictor.set_model(settings.stardist_model)
    # pipeline._myosam_predictor.set_model(settings.myosam_model)
    yield
    # Clean up and closing redis
    # del pipeline
    # await redis.close()
    print("Stopping application")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get_config/", response_model=Config)
async def get_config():
    """Get the configuration of the pipeline."""
    return Config(measure_unit=1.0)


@app.get("/redis_status/")
async def status():
    """check status of the redis connection"""
    if random.randint(0, 1):
        return {"status": True}
    else:
        return {"status": False}
