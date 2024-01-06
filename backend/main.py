from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, BackgroundTasks, Request, Depends
from fastapi.responses import Response
from redis import asyncio as aioredis  # type: ignore

from myo_sam.inference.pipeline import Pipeline

from functools import lru_cache

from .config import Settings

settings = Settings(_env_file=".env", _env_file_encoding="utf-8")
pipeline: Pipeline = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    # Clean up
    del pipeline
    await redis.close()
    print("Stopping application")


app = FastAPI(lifespan=lifespan)


@lru_cache  # Work on single connection
async def setup_redis() -> aioredis.Redis:
    # Load models, connect to database, etc.
    try:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        yield redis
    except Exception as e:
        print(f"Failed establishing connection to redis: {e}")
    finally:
        # Clean up
        print("Stopping application")
        await redis.close()


async def return_redis_status(redis: aioredis.Redis) -> bool:
    try:
        status = await redis.ping()
        return status
    except Exception as e:
        print(f"Failed establishing connection to redis: {e}")
        return False


async def clear_cache(redis: aioredis.Redis) -> None:
    # Clear cache in redis
    await redis.delete()


@app.get("/")
def root(request: Request):
    return {"message": "Hello World"}


@app.exception_handler(404)
async def error_handler(request: Request, exc: Exception) -> Response:
    """Redirects request and exception to /error"""
    return Response(b"not found", status_code=404)
    # return RedirectResponse(url="/", status_code=404)


@app.get("/configure")
async def configure(request: Request) -> None:
    request_data = await request.form()
    settings.update_myosam_config(request_data.get("myosam_config"))
    settings.update_measure_unit(request_data.get("measure_unit"))


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


async def set_cache(
    data: dict, redis: Annotated[aioredis.Redis, Depends(setup_redis)]
) -> None:
    # Set cache in redis for 1 hour
    await redis.set("key", "value", ex=3600)
