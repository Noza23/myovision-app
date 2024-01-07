from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import (
    FastAPI,
    Depends,
    BackgroundTasks,
    Request,
    UploadFile,
    File,
)

from fastapi.responses import Response  # RedirectResponse
from redis import asyncio as aioredis  # type: ignore

from myo_sam.inference.pipeline import Pipeline

from functools import lru_cache

from .models import Settings, REDIS_KEYS

settings = Settings(_env_file=".env", _env_file_encoding="utf-8")
pipeline: Pipeline = None
redis_keys = REDIS_KEYS(prefix="myo_sam")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan of application."""
    print("Starting application")
    # Redis connection
    print("Connecting to redis")
    # redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    # Loading models
    print("Loading models")
    global pipeline
    pipeline = Pipeline()
    # pipeline._stardist_predictor.set_model(settings.STARDIST_MODEL)
    # pipeline._myosam_predictor.set_model(settings.MYOSAM_MODEL)
    yield
    # Clean up and closing redis
    del pipeline
    # await redis.close()
    print("Stopping application")


app = FastAPI(lifespan=lifespan)  # lifespan=lifespan


def get_pipeline_instance() -> Pipeline:
    """Each user gets new pipeline instance, where models are shared."""
    return pipeline.model_copy()


@lru_cache  # Work on single connection
async def setup_redis() -> aioredis.Redis:
    """Redis connection is single and shared across all users."""
    try:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    except Exception as e:
        print(f"Failed establishing connection to redis: {e}")
    return redis


async def set_cache(key: str, value: str, redis: aioredis.Redis) -> None:
    """cache single item"""
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


@app.get("/status")
async def status(redis: Annotated[aioredis.Redis, Depends(setup_redis)]):
    """check status of the redis connection"""
    try:
        status = await redis.ping()
        return {"status": status}
    except Exception as e:
        print(f"Failed establishing connection to redis: {e}")
        return {"status": False}


@app.post("/run")  # response_model=RedirectResponse
async def run(
    background_tasks: BackgroundTasks,
    redis: Annotated[aioredis.Redis, Depends(setup_redis)],
    pipeline: Annotated[Pipeline, Depends(get_pipeline_instance)],
    keys: Annotated[REDIS_KEYS, Depends(REDIS_KEYS)],
    myotube: UploadFile = File(None),
    nuclei: UploadFile = File(None),
):
    """Run the pipeline"""
    myo_cache, nucl_cache = None, None

    if myotube.filename:
        pipeline.set_nuclei_image(await myotube.read())
        if await is_cached(keys.myotube_key(pipeline.myotube_hash), redis):
            myo_cache = await redis.get(
                keys.myotube_key(pipeline.myotube_hash)
            )

    if nuclei.filename:
        pipeline.set_myotube_image(await nuclei.read())
        if await is_cached(keys.nuclei_key(pipeline.nuclei_hash), redis):
            nucl_cache = await redis.get(keys.nuclei_key(pipeline.nuclei_hash))

    result = pipeline.execute(
        myotubes_cached=myo_cache, nucleis_cached=nucl_cache
    )

    if pipeline.myotube_image:
        background_tasks.add_task(
            set_cache,
            keys.myotube_key(pipeline.myotube_hash),
            result.information_metrics.myotubes.model_dump_json(),
            redis,
        )

    if pipeline.nuclei_image:
        background_tasks.add_task(
            set_cache,
            keys.nuclei_key(pipeline.nuclei_hash),
            result.information_metrics.nucleis.model_dump_json(),
            redis,
        )

    return result.information_metrics.model_dump_json()


@app.exception_handler(404)
async def error_handler(request: Request, exc: Exception) -> Response:
    """Redirects request and exception to /error"""
    return Response(b"not found", status_code=404)
    # return RedirectResponse(url="/", status_code=404)
