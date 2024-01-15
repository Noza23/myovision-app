from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated
import json

from fastapi import (
    FastAPI,
    Depends,
    BackgroundTasks,
    Request,
    UploadFile,
    File,
    HTTPException,
)
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as aioredis  # type: ignore

from myo_sam.inference.pipeline import Pipeline
from myo_sam.inference.models.information import InformationMetrics


from .models import (
    Settings,
    REDIS_KEYS,
    Config,
    State,
    ValidationResponse,
    InferenceResponse,
)
from .utils import get_fp


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


@lru_cache  # Work on single connection
async def setup_redis() -> aioredis.Redis:
    """Redis connection is single and shared across all users."""
    try:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    except Exception as e:
        print(f"Failed establishing connection to redis: {e}")
    return redis


async def set_cache(mapping: dict[str, str], redis: aioredis.Redis) -> None:
    """cache multiple items"""
    await redis.mset(mapping)
    await redis.bgsave()


async def is_cached(key: str, redis: aioredis.Redis) -> bool:
    """check if key is cached"""
    return await redis.exists(key) > 0


async def clear_cache(redis: aioredis.Redis, key: str) -> None:
    """clear cache for a single key"""
    await redis.delete(key)


@app.get("/get_config/", response_model=Config)
async def get_config():
    """Get the configuration of the pipeline."""
    return Config()


@app.post("/validation/", response_model=ValidationResponse)
async def run_validation(
    image: UploadFile,
    config: Config,
    background_tasks: BackgroundTasks,
    redis: Annotated[aioredis.Redis, Depends(setup_redis)],
    pipeline: Annotated[Pipeline, Depends(get_pipeline_instance)],
    keys: Annotated[REDIS_KEYS, Depends(REDIS_KEYS)],
):
    """Run the pipeline in validation mode."""
    # Set the image
    pipeline.set_myotube_image(await image.read(), image.filename)
    img_hash = pipeline.myotube_hash

    if await is_cached(keys.result_key(img_hash), redis):
        # Case when image is cached
        myos = json.loads(await redis.get(keys.result_key(img_hash)))
        state = State.model_validate_json(
            await redis.get(keys.state_key(img_hash))
        )
        path = await redis.get(keys.image_path_key(img_hash))
        if not path:
            # path might be cleaned by regular image cleaning
            path = get_fp(settings.images_dir)
            _ = pipeline.save_myotube_image(path)
            background_tasks.add_task(
                set_cache, {keys.image_path_key(img_hash): path}, redis
            )
    else:
        # Case when image is not cached
        state = State()
        pipeline._myosam_predictor.update_amg_config(config.amg_config)
        pipeline.set_measure_unit(config.general_config.measure_unit)
        myos = pipeline.execute().information_metrics.myotubes
        path = get_fp(settings.images_dir)
        pipeline.save_myotube_image(path)

        background_tasks.add_task(
            set_cache,
            {
                keys.result_key(img_hash): myos.model_dump_json(),
                keys.image_path_key(img_hash): path,
                keys.state_key(img_hash): state.model_dump_json(),
            },
            redis,
        )

    return ValidationResponse(
        roi_coords=[myo.roi_coords for myo in myos],
        state=state,
        hash_str=img_hash,
    )


@app.post("/inference/", response_model=InferenceResponse)
async def run_inference(
    config: Config,
    background_tasks: BackgroundTasks,
    redis: Annotated[aioredis.Redis, Depends(setup_redis)],
    pipeline: Annotated[Pipeline, Depends(get_pipeline_instance)],
    keys: Annotated[REDIS_KEYS, Depends(REDIS_KEYS)],
    myotube: UploadFile = File(None),
    nuclei: UploadFile = File(None),
):
    """Run the pipeline in inference mode."""
    if not myotube.filename and not nuclei.filename:
        raise HTTPException(
            status_code=400,
            detail="Either myotube or nuclei image must be provided.",
        )

    myo_cache, nucl_cache = None, None
    if myotube.filename:
        pipeline.set_myotube_image(await myotube.read(), myotube.filename)
        img_hash = pipeline.myotube_hash
        if await is_cached(keys.result_key(img_hash), redis):
            myo_cache = await redis.get(keys.result_key(img_hash))
            path = await redis.get(keys.image_path_key(img_hash))
            if not path:
                # path might be cleaned by regular image cleaning
                path = get_fp(settings.images_dir)
                _ = pipeline.save_myotube_image(path)
                background_tasks.add_task(
                    set_cache, {keys.image_path_key(img_hash): path}, redis
                )
    if nuclei.filename:
        pipeline.set_nuclei_image(await nuclei.read(), nuclei.filename)
        sec_img_hash = pipeline.nuclei_hash
        if await is_cached(keys.result_key(sec_img_hash), redis):
            nucl_cache = await redis.get(keys.result_key(sec_img_hash))

    # Execute Pipeline
    pipeline._myosam_predictor.update_amg_config(config.amg_config)
    pipeline.set_measure_unit(config.general_config.measure_unit)
    result = pipeline.execute(myo_cache, nucl_cache).information_metrics
    myos, nucls = result.myotubes, result.nucleis

    if myotube.filename and not myo_cache:
        background_tasks.add_task(
            set_cache,
            {keys.result_key(sec_img_hash): myos.model_dump_json()},
        )

    if nuclei.filename and not nucl_cache:
        background_tasks.add_task(
            set_cache,
            {keys.result_key(sec_img_hash): nucls.model_dump_json()},
        )

    # Overlay contours on main image
    img_drawn = pipeline.draw_contours_on_myotube_image(myos, nucls)
    path = get_fp(settings.images_dir)
    pipeline.save_myotube_image(path, img_drawn)
    return InferenceResponse(
        iamge_path=path, image_hash=img_hash, secondary_image_hash=sec_img_hash
    )


@app.get("/redis_status/")
async def status(redis: Annotated[aioredis.Redis, Depends(setup_redis)]):
    """check status of the redis connection"""
    try:
        status = await redis.ping()
        return {"status": status}
    except Exception as e:
        print(f"Failed establishing connection to redis: {e}")
        return {"status": False}


@app.get("/adjust_unit/", response_model=InformationMetrics)
def adjust_unit(metrics: InformationMetrics, mu: float):
    """Adjust unit of the metrics"""
    return metrics.adjust_measure_unit(mu)


@app.exception_handler(404)
async def error_handler(request: Request, exc: Exception) -> Response:
    """Redirects request and exception to /error"""
    return Response(b"not found", status_code=404)
    # return RedirectResponse(url="/", status_code=404)
