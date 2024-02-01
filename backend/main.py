from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Union, Optional, Any

from pydantic import ValidationError
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
from myo_sam.inference.models.base import Myotubes, MyoObjects, NucleiClusters
from myo_sam.inference.models.information import InformationMetrics
from myo_sam.inference.predictors.utils import invert_image

from .models import (
    Settings,
    REDIS_KEYS,
    Config,
    State,
    ValidationResponse,
    InferenceResponse,
    Point,
)
from .utils import get_fp, preprocess_ws_resp


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
) -> dict[str, bool]:
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
async def get_config_schema() -> dict[str, Any]:
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
    redis: Annotated[Union[aioredis.Redis, None], Depends(setup_redis)],
    pipeline: Annotated[Pipeline, Depends(get_pipeline_instance)],
    myotube: UploadFile = File(None),
    nuclei: UploadFile = File(None),
):
    """Run the pipeline in inference mode."""
    if not redis:
        raise HTTPException(
            status_code=400,
            detail="Redis connection is not available.",
        )
    if not hasattr(myotube, "filename") and not hasattr(nuclei, "filename"):
        raise HTTPException(
            status_code=400,
            detail="Either myotube or nuclei image must be provided.",
        )

    if hasattr(myotube, "filename"):
        pipeline.set_myotube_image(await myotube.read(), myotube.filename)
        img_hash = pipeline.myotube_hash
        if await redis.exists(redis_keys.result_key(img_hash)):
            myo_cache = await redis.get(redis_keys.result_key(img_hash))
        else:
            myo_cache = None
    else:
        img_hash = None

    if hasattr(nuclei, "filename"):
        pipeline.set_nuclei_image(await nuclei.read(), nuclei.filename)
        img_sec_hash = pipeline.nuclei_hash
        if await redis.exists(redis_keys.result_key(img_sec_hash)):
            nucl_cache = await redis.get(redis_keys.result_key(img_sec_hash))
        else:
            nucl_cache = None
    else:
        img_sec_hash = None

    # Execute Pipeline
    pipeline._myosam_predictor.update_amg_config(config.amg_config)
    pipeline.set_measure_unit(config.general_config.measure_unit)
    result = pipeline.execute(myo_cache, nucl_cache).information_metrics
    myos, nucls = result.myotubes, result.nucleis

    if hasattr(myotube, "filename") and not myo_cache:
        background_tasks.add_task(
            redis.set, redis_keys.result_key(img_hash), myos.model_dump_json()
        )

    if hasattr(myotube, "filename") and not nucl_cache:
        background_tasks.add_task(
            redis.set,
            redis_keys.result_key(img_sec_hash),
            nucls.model_dump_json(),
        )

    # Overlay contours on main image
    img_drawn = pipeline.draw_contours_on_myotube_image(myos, nucls)
    path = get_fp(settings.images_dir)
    pipeline.save_myotube_image(path, img_drawn)

    return InferenceResponse(
        image_path=path, image_hash=img_hash, image_secondary_hash=img_sec_hash
    )


@app.websocket("/inference/{hash_str}/{sec_hash_str}")
async def inference_ws(
    websocket: WebSocket,
    hash_str: Optional[str],
    sec_hash_str: Optional[str],
    redis: Annotated[Union[aioredis.Redis, None], Depends(setup_redis)],
) -> None:
    """Websocket for inference mode."""
    if not redis:
        raise HTTPException(
            status_code=400,
            detail="Redis connection is not available.",
        )
    if not hash_str and not sec_hash_str:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Hash string missing.",
        )

    if hash_str:
        myotubes = Myotubes.model_validate_json(
            await redis.get(redis_keys.result_key(hash_str))
        )
    else:
        myotubes = None

    if sec_hash_str:
        nucleis = MyoObjects.model_validate_json(
            await redis.get(redis_keys.result_key(sec_hash_str))
        )
    else:
        nucleis = None

    if isinstance(myotubes, Myotubes) and isinstance(nucleis, MyoObjects):
        clusters = NucleiClusters.compute_clusters(nucleis)
    else:
        clusters = NucleiClusters()

    info_data = InformationMetrics(myotubes, nucleis, clusters)
    info_data.model_dump_json(
        exclude={"myotubes", "nucleis", "nuclei_clusters"}
    )

    await websocket.accept()
    # Initial websocket sends general information like total number area ...
    # websockets awaits on (x, y) and sends back instance specific information
    while True:
        await websocket.send_json(
            {
                "info_data": info_data.model_dump(
                    exclude={"myotubes", "nucleis", "nuclei_clusters"}
                )
            }
        )
        if len(myotubes) + len(nucleis) == 0:
            await websocket.close()
            break

        # Wating for response from front {x: int, y: int}
        data = await websocket.receive_json()

        try:
            point = Point.model_validate(data)
        except ValidationError:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Invalid data received.",
            )
        myo = myotubes.get_instance_by_point(point.to_tuple())
        if myo:
            clusts = clusters.get_clusters_by_myotube_id(myo.identifier)
            resp = {
                "info_data": {
                    "myotube": preprocess_ws_resp(myo.model_dump()),
                    "clusters": [clust.model_dump() for clust in clusts],
                }
            }
        else:
            resp = {"info_data": {"myotube": None, "clusters": None}}
        await websocket.send_json(resp)


@app.get("/get_contours/{hash_str}")
async def get_contours(
    hash_str: str,
    redis: Annotated[Union[aioredis.Redis, None], Depends(setup_redis)],
) -> dict[str, list[list[list[int]]]]:
    """Get the contours for specific image."""
    if not redis:
        raise HTTPException(
            status_code=400,
            detail="Redis connection is not available.",
        )

    objs = MyoObjects.model_validate(
        await redis.get(redis_keys.result_key(hash_str))
    )
    if not objs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No contours found for this image.",
        )
    return {"roi_coords": [x.roi_coords for x in objs]}
