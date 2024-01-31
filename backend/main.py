from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Union

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
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as aioredis  # type: ignore

from myo_sam.inference.pipeline import Pipeline
from myo_sam.inference.models.base import Myotubes, MyoObjects
from myo_sam.inference.predictors.utils import invert_image

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
origins = ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan of application."""
    print("> Starting Application...")

    print("> Starting Redis...")
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    print("> Loading models...")
    global pipeline
    pipeline = Pipeline()
    pipeline._stardist_predictor.set_model(settings.stardist_model)
    pipeline._myosam_predictor.set_model(settings.myosam_model)
    yield

    print("> Shutting down...")
    del pipeline
    if redis:
        await redis.close()


app = FastAPI(lifespan=lifespan, title="MyoVision API", version="0.1.0")
app.mount("/images", StaticFiles(directory=settings.images_dir), name="images")

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
def setup_redis() -> Union[aioredis.Redis, None]:
    """Redis connection is single and shared across all users."""
    try:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    except Exception as e:
        print(f"! Failed establishing connection to redis: {e}")
        redis = None
    return redis


@app.get("/redis_status/")
async def redis_status(
    redis: Annotated[Union[aioredis.Redis, None], Depends(setup_redis)],
):
    """check status of the redis connection"""
    if not redis:
        return {"status": False}
    try:
        status = await redis.ping()
        return {"status": status}
    except Exception as e:
        print(f"! Failed establishing connection to redis: {e}")
        return {"status": False}


@app.get("/get_config_schema/")
async def get_config_schema() -> dict:
    """Get the configuration of the pipeline."""
    return Config(measure_unit=1.0).model_json_schema()["$defs"]


@app.post("/validation/", response_model=ValidationResponse)
async def run_validation(
    image: UploadFile,
    config: Config,
    background_tasks: BackgroundTasks,
    redis: Annotated[Union[aioredis.Redis, None], Depends(setup_redis)],
    pipeline: Annotated[Pipeline, Depends(get_pipeline_instance)],
):
    """Run the pipeline in validation mode."""
    if not redis:
        raise HTTPException(
            status_code=400,
            detail="Redis connection is not available.",
        )
    # Set the image
    pipeline.set_myotube_image(await image.read(), image.filename)

    if any(pipeline.myotube_image_np.shape > 2048):
        print("> Image is too large, cropping to 2048x2048")
        pipeline.set_myotube_image(
            pipeline.myotube_image_np[:2048, :2048, :],
            image.filename,
        )
    img_hash = pipeline.myotube_hash

    img_to_send = pipeline.myotube_image_np.copy()
    if config.general_config.invert_image:
        img_to_send = invert_image(img_to_send)

    if await redis.exists(redis_keys.result_key(img_hash)):
        # Case when image is cached
        myos = Myotubes.model_validate_json(
            await redis.get(redis_keys.result_key(img_hash))
        )
        if myos[0].measure_unit != config.general_config.measure_unit:
            myos.adjust_measure_unit(config.general_config.measure_unit)
            await redis.set(
                redis_keys.result_key(img_hash), myos.model_dump_json()
            )
        state = State.model_validate_json(
            await redis.get(redis_keys.state_key(img_hash))
        )
        if state.valid:
            img_to_send = pipeline.draw_contours(
                img_to_send,
                [myos[i].roi_coords for i in state.valid],
            )
    else:
        # Case when image is not cached
        state = State()
        pipeline._myosam_predictor.update_amg_config(config.amg_config)
        pipeline.set_measure_unit(config.general_config.measure_unit)
        myos = pipeline.execute().information_metrics.myotubes
        background_tasks.add_task(
            redis.mset,
            {
                redis_keys.result_key(img_hash): myos.model_dump_json(),
                redis_keys.state_key(img_hash): state.model_dump_json(),
            },
        )
    path = get_fp(settings.images_dir, img_to_send)
    pipeline.save_image(path, img_to_send)
    return ValidationResponse(image_hash=img_hash, image_path=path)


@app.websocket("/validation/{hash_str}")
async def validation_ws(
    websocket: WebSocket,
    hash_str: str,
    background_tasks: BackgroundTasks,
    redis: Annotated[Union[aioredis.Redis, None], Depends(setup_redis)],
):
    """Websocket for validation mode."""
    if not redis:
        raise HTTPException(
            status_code=400,
            detail="Redis connection is not available.",
        )

    await websocket.accept()

    mo = MyoObjects.model_validate_json(
        await redis.get(redis_keys.result_key(hash_str))
    )
    state = State.model_validate_json(
        await redis.get(redis_keys.state_key(hash_str))
    )
    if state.done:
        await websocket.send_json(
            {
                "roi_coords": None,
                "contour_id": len(mo),
                "total": len(mo),
            }
        )
        await websocket.close()
        return

    i = state.get_next()
    await websocket.send_json(
        {
            "roi_coords": mo[i].roi_coords,
            "contour_id": i + 1,
            "total": len(mo),
        }
    )

    while True:
        if len(mo) == i + 1:
            state.done = True
            websocket.send_json(
                {
                    "roi_coords": None,
                    "contour_id": i + 1,
                    "total": len(mo),
                }
            )
            background_tasks.add_task(
                redis.set,
                redis_keys.state_key(hash_str),
                state.model_dump_json(),
            )
            await websocket.close()
            break

        else:
            # Invalid = 0, Valid = 1, Skip = 2, Undo = -1
            data = int(await websocket.receive_text())
            if data == 0:
                state.invalid.add(i)
            elif data == 1:
                state.valid.add(i)
            elif data == 2:
                _ = mo.move_object_to_end(i)
                background_tasks.add_task(
                    redis.set,
                    redis_keys.result_key(hash_str),
                    mo.model_dump_json(),
                )
            elif data == -1:
                state.valid.discard(i)
                state.invalid.discard(i)
            else:
                raise WebSocketException(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Invalid data received.",
                )
            background_tasks.add_task(
                redis.set,
                redis_keys.state_key(hash_str),
                state.model_dump_json(),
            )

            # Send next contour
            step = data != 2 if data != -1 else -1
            i = min(max(i + step, 0), len(mo) - 1)
            await websocket.send_json(
                {
                    "roi_coords": mo[i].roi_coords,
                    "contour_id": i + 1,
                    "total": len(mo),
                }
            )


@app.post("/inference/", response_model=InferenceResponse)
async def run_inference(
    config: Config,
    background_tasks: BackgroundTasks,
    redis: Annotated[aioredis.Redis, Depends(setup_redis)],
    pipeline: Annotated[Pipeline, Depends(get_pipeline_instance)],
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
        if await redis.exists(redis_keys.result_key(img_hash)):
            myo_cache = await redis.get(redis_keys.result_key(img_hash))
            path = await redis.get(redis_keys.image_path_key(img_hash))
            if not path:
                # path might be cleaned by regular image cleaning
                path = get_fp(settings.images_dir)
                _ = pipeline.save_myotube_image(path)
                background_tasks.add_task(
                    redis.set, redis_keys.image_path_key(img_hash), path
                )
    if nuclei.filename:
        pipeline.set_nuclei_image(await nuclei.read(), nuclei.filename)
        sec_img_hash = pipeline.nuclei_hash
        if await redis.exists(redis_keys.result_key(sec_img_hash)):
            nucl_cache = await redis.get(redis_keys.result_key(sec_img_hash))

    # Execute Pipeline
    pipeline._myosam_predictor.update_amg_config(config.amg_config)
    pipeline.set_measure_unit(config.general_config.measure_unit)
    result = pipeline.execute(myo_cache, nucl_cache).information_metrics
    myos, nucls = result.myotubes, result.nucleis

    if myotube.filename and not myo_cache:
        background_tasks.add_task(
            redis.set, redis_keys.result_key(img_hash), myos.model_dump_json()
        )

    if nuclei.filename and not nucl_cache:
        background_tasks.add_task(
            redis.set,
            redis_keys.result_key(sec_img_hash),
            nucls.model_dump_json(),
        )

    # Overlay contours on main image
    img_drawn = pipeline.draw_contours_on_myotube_image(myos, nucls)
    path = get_fp(settings.images_dir)
    pipeline.save_myotube_image(path, img_drawn)

    return InferenceResponse(
        iamge_path=path,
        image_hash=img_hash if myotube.filename else None,
        secondary_image_hash=sec_img_hash if nuclei.filename else None,
    )
