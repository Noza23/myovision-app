from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, BackgroundTasks, Request, WebSocket, Depends
from fastapi.responses import Response
from redis import asyncio as aioredis  # type: ignore

from myo_sam.inference.pipeline import Pipeline

from functools import lru_cache

from .config import Settings, REDIS_KEYS

settings = Settings(_env_file=".env", _env_file_encoding="utf-8")
pipeline: Pipeline = None
redis_keys = REDIS_KEYS(prefix="myo_sam")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan of application."""
    print("Starting application")
    # Redis connection
    print("Connecting to redis")
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    # Loading models
    print("Loading models")
    pipeline = Pipeline()
    pipeline._stardist_predictor.set_model(settings.STARDIST_MODEL)
    pipeline._myosam_predictor.set_model(settings.MYOSAM_MODEL)
    yield
    # Clean up and closing redis
    del pipeline
    await redis.close()
    print("Stopping application")


app = FastAPI(lifespan=lifespan)


def get_pipeline_instance() -> Pipeline:
    """Each user gets new pipeline instance, where models are shared."""
    return pipeline.model_copy()


@lru_cache  # Work on single connection
async def setup_redis() -> aioredis.Redis:
    """Redis connection is single and shared across all users."""
    try:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        redis.config_set("appendonly", "yes")
    except Exception as e:
        print(f"Failed establishing connection to redis: {e}")
    return redis


async def return_redis_status(redis: aioredis.Redis) -> bool:
    try:
        status = await redis.ping()
        return status
    except Exception as e:
        print(f"Failed establishing connection to redis: {e}")
        return False


async def set_cache(key: str, value: str, redis: aioredis.Redis) -> None:
    """cache single item for 1 hour"""
    await redis.set(key, value)
    await redis.bgsave()


async def is_cached(key: str, redis: aioredis.Redis) -> bool:
    """check if key is cached"""
    return await redis.exists(key) > 0


async def clear_cache(redis: aioredis.Redis, key: str) -> None:
    """clear cache for a single key"""
    await redis.delete(key)


@app.get("/")
def root(request: Request):
    return {"message": "Hello World"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    pass


@app.get("/status")
async def status(
    redis: Annotated[aioredis.Redis, Depends(setup_redis)],
):
    # Check if redis is up
    status = await return_redis_status(redis)
    return {"status": status}


@app.get("/run")
async def run(
    data: dict,
    background_tasks: BackgroundTasks,
    redis: Annotated[aioredis.Redis, Depends(setup_redis)],
):
    # Run the pipeline
    # cache results in redis

    return {"message": "Hello World"}


@app.exception_handler(404)
async def error_handler(request: Request, exc: Exception) -> Response:
    """Redirects request and exception to /error"""
    return Response(b"not found", status_code=404)
    # return RedirectResponse(url="/", status_code=404)
