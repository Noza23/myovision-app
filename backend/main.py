from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated
import json

from fastapi import (
    FastAPI,
    Depends,
    BackgroundTasks,
    UploadFile,
    File,
    HTTPException,
    WebSocket,
    WebSocketException,
    status,
)
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as aioredis  # type: ignore

from myo_sam.inference.pipeline import Pipeline
from myo_sam.inference.models.base import Nucleis, Myotubes, MyoObjects
from myo_sam.inference.utils import hash_bytes

from .models import (
    Settings,
    REDIS_KEYS,
    Config,
    State,
    ValidationResponse,
    InferenceResponse,
)
from .utils import get_fp
from pydantic import ValidationError


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


app = FastAPI(lifespan=lifespan, title="MyoVision API", version="0.1.0")

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
        myos = Myotubes.model_validate_json(
            await redis.get(keys.result_key(img_hash))
        )
        if not myos:
            raise HTTPException(
                status_code=404,
                detail="myotubes not found for the given hash.",
            )

        if myos[0].measure_unit != config.general_config.measure_unit:
            myos.adjust_measure_unit(config.general_config.measure_unit)
            await redis.set(keys.result_key(img_hash), myos.model_dump_json())

        state = State.model_validate_json(
            await redis.get(keys.state_key(img_hash))
        )
        img_drawn = pipeline.draw_contours_on_myotube_image(
            myos.filter_by_ids(state.valid), thickness=2
        )
        img_drawn_hash = hash_bytes(img_drawn)
        path = await redis.get(keys.image_path_key(img_drawn_hash))
        if not path:
            # path might be cleaned by regular image cleaning
            path = get_fp(settings.images_dir)
            pipeline.save_myotube_image(path, img_drawn)
            background_tasks.add_task(
                set_cache, {keys.image_path_key(img_drawn_hash): path}, redis
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

    return ValidationResponse(hash_str=img_hash, image_path=path)


@app.websocket("/validation/{hash_str}/")
async def validation_ws(
    websocket: WebSocket,
    hash_str: str,
    background_tasks: BackgroundTasks,
    keys: Annotated[REDIS_KEYS, Depends(REDIS_KEYS)],
    redis: Annotated[aioredis.Redis, Depends(setup_redis)],
):
    """Websocket for validation mode."""
    await websocket.accept()
    mo = MyoObjects.model_validate_json(
        await redis.get(keys.result_key(hash_str))
    )
    state = State.model_validate_json(
        await redis.get(keys.state_key(hash_str))
    )

    if state.done:
        await websocket.send_text("done")
    i = state.get_next()
    # Starting Contour send on connection openning
    await websocket.send_json(
        {"roi_coords": mo[i].roi_coords, "contour_id": i}
    )
    while True:
        if len(mo) == i:
            state.done = True
            websocket.send_text("done")
        # Wating for response from front
        data = int(await websocket.receive_text())
        # Invalid = 0, Valid = 1, Skip = 2, Undo = -1
        assert data in (0, 1, 2, -1)
        if data == 0:
            # Invalid contour
            state.invalid.add(i)
        elif data == 1:
            # Valid contour
            state.valid.add(i)
        elif data == 2:
            # Skip contour: move to end and recache result
            _ = mo.move_object_to_end(i)
            await set_cache(
                {keys.result_key(hash_str): mo.model_dump_json()}, redis
            )
        elif data == -1:
            # Undo contour
            state.valid.discard(i)
            state.invalid.discard(i)
        else:
            raise WebSocketException(
                status_code=status.WS_1008_POLICY_VIOLATION,
                detail="Invalid data received.",
            )
        # Update state in cache
        background_tasks.add_task(
            set_cache,
            {keys.state_key(hash_str): state.model_dump_json()},
            redis,
        )
        # Send next contour
        step = data != 2 if data != -1 else -1
        await websocket.send_json(
            {"roi_coords": mo[i + step].roi_coords, "contour_id": i + step}
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
async def redis_status(redis: Annotated[aioredis.Redis, Depends(setup_redis)]):
    """check status of the redis connection"""
    try:
        status = await redis.ping()
        return {"status": status}
    except Exception as e:
        print(f"Failed establishing connection to redis: {e}")
        return {"status": False}


@app.get("/adjust_unit/{hash_str}/{mu}")
async def adjust_unit(
    hash_str: str,
    mu: float,
    keys: Annotated[REDIS_KEYS, Depends(REDIS_KEYS)],
    redis: Annotated[aioredis.Redis, Depends(setup_redis)],
):
    """Adjust unit of the metrics"""
    objs = json.loads(await redis.get(keys.result_key(hash_str)))
    if not objs["myo_objects"]:
        raise HTTPException(
            status_code=404, detail="myo_objects not found for the given hash."
        )
    try:
        objs = Myotubes.model_validate(objs)
    except ValidationError:
        objs = Nucleis.model_validate(objs)
    objs.adjust_measure_unit(mu)
    await redis.set(keys.result_key(hash_str), objs.model_dump_json())
    return Response(status_code=200)
