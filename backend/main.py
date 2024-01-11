from contextlib import asynccontextmanager

# from typing import Annotated
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# from redis import asyncio as aioredis  # type: ignore

from myo_sam.inference.pipeline import Pipeline
from myo_sam.inference.models.information import InformationMetrics

# from functools import lru_cache

from .models import Settings, REDIS_KEYS, Config
import json
import random

settings = Settings(_env_file=".env", _env_file_encoding="utf-8")
pipeline: Pipeline = None
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
    global pipeline
    pipeline = Pipeline()
    # pipeline._stardist_predictor.set_model(settings.stardist_model)
    # pipeline._myosam_predictor.set_model(settings.myosam_model)
    yield
    # Clean up and closing redis
    del pipeline
    # await redis.close()
    print("Stopping application")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline_instance() -> Pipeline:
    """Each user gets new pipeline instance, where models are shared."""
    return pipeline.model_copy()


@app.get("/get_config/", response_model=Config)
async def get_config():
    """Get the configuration of the pipeline."""
    return Config(
        amg_config=pipeline._myosam_predictor.amg_config,
        measure_unit=pipeline.measure_unit,
    )


@app.post("/run/", response_model=InformationMetrics)
async def run(
    config: Config,
    myotube: UploadFile = File(None),
    nuclei: UploadFile = File(None),
):
    if myotube:
        print("myotube: ", myotube.filename)
    if nuclei:
        print("nuclei: ", nuclei.filename)
    print("form: ", config.model_dump_json())
    info = InformationMetrics.model_validate(
        json.load(open("data/info_data.json"))
    )
    info.myotubes = info.myotubes[:2]
    info.nucleis = info.nucleis[:2]
    return info


@app.get("/redis_status/")
async def status():
    """check status of the redis connection"""
    if random.randint(0, 1):
        return {"status": True}
    else:
        return {"status": False}


@app.exception_handler(404)
async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Redirects request and exception to /error"""
    return JSONResponse(
        status_code=404,
        content={"message": "Not found"},
    )


# @lru_cache  # Work on single connection
# async def setup_redis() -> None:
#     """Redis connection is single and shared across all users."""
#     print("Fake connection to redis")
#     try:
#         redis = aioredis.from_url(settings.redis_url, decode_responses=True)
#     except Exception as e:
#         print(f"Failed establishing connection to redis: {e}")
#     return redis


# async def set_cache(key: str, value: str, redis: aioredis.Redis) -> None:
#     """cache single item"""
#     await redis.set(key, value)
#     await redis.bgsave()


# async def is_cached(key: str, redis: aioredis.Redis) -> bool:
#     """check if key is cached"""
#     return await redis.exists(key) > 0


# async def clear_cache(redis: aioredis.Redis, key: str) -> None:
#     """clear cache for a single key"""
#     await redis.delete(key)

# @app.post("/run/", response_model=InformationMetrics)
# async def run(
#     background_tasks: BackgroundTasks,
#     redis: Annotated[aioredis.Redis, Depends(setup_redis)],
#     pipeline: Annotated[Pipeline, Depends(get_pipeline_instance)],
#     keys: Annotated[REDIS_KEYS, Depends(REDIS_KEYS)],
#     myotube: UploadFile = File(None),
#     nuclei: UploadFile = File(None),
# ):
#     """Run the pipeline"""
#     myo_cache, nucl_cache = None, None
#     if myotube.filename:
#         pipeline.set_nuclei_image(await myotube.read(), myotube.filename)
#         if await is_cached(keys.myotube_key(pipeline.myotube_hash), redis):
#             myo_cache = await redis.get(
#                 keys.myotube_key(pipeline.myotube_hash)
#             )

#     if nuclei.filename:
#         pipeline.set_myotube_image(await nuclei.read(), nuclei.filename)
#         if await is_cached(keys.nuclei_key(pipeline.nuclei_hash), redis):
#             nucl_cache = await redis.get(keys.nuclei_key(pipeline.nuclei_hash))

#     result = pipeline.execute(myo_cache, nucl_cache)

#     if pipeline.myotube_image:
#         background_tasks.add_task(
#             set_cache,
#             keys.myotube_key(pipeline.myotube_hash),
#             result.information_metrics.myotubes.model_dump_json(),
#             redis,
#         )

#     if pipeline.nuclei_image:
#         background_tasks.add_task(
#             set_cache,
#             keys.nuclei_key(pipeline.nuclei_hash),
#             result.information_metrics.nucleis.model_dump_json(),
#             redis,
#         )
#     return result.information_metrics.model_dump_json()


# @app.get("/status/")
# async def status(redis: Annotated[aioredis.Redis, Depends(setup_redis)]):
#     """check status of the redis connection"""
#     try:
#         status = await redis.ping()
#         return {"status": status}
#     except Exception as e:
#         print(f"Failed establishing connection to redis: {e}")
#         return {"status": False}


# @app.get("/adjust_unit/", response_model=InformationMetrics)
# def adjust_unit(metrics: InformationMetrics, mu: float):
#     """Adjust unit of the metrics"""
#     return metrics.adjust_measure_unit(mu)
