from typing import Any, Union

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
)
from myosam.inference.models.base import Myotubes, NucleiClusters, Nucleis
from myosam.inference.pipeline import Pipeline
from pydantic import ValidationError
from redis import asyncio as aioredis

from backend import STATIC_IMAGES_DIR, RedisKeys
from backend.dependancies import get_pipeline_instance, setup_redis
from backend.models import Config, InferenceResponse, Point
from backend.utils import get_fp, preprocess_ws_resp

router = APIRouter(
    prefix="/inference",
    tags=["inference"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=InferenceResponse)
async def run_inference(
    config: Config,
    myotube: UploadFile = File(None),
    nuclei: UploadFile = File(None),
    redis: aioredis.Redis = Depends(setup_redis),
    pipeline: Pipeline = Depends(get_pipeline_instance),
):
    """Run the pipeline in inference mode."""
    myo_cache, nucl_cache = None, None
    pipeline.set_myotube_image(await myotube.read(), myotube.filename)
    img_hash = pipeline.myotube_hash()
    if await redis.exists(RedisKeys.result_key(img_hash)):
        myo_cache = await redis.get(RedisKeys.result_key(img_hash))

    if hasattr(nuclei, "filename"):
        pipeline.set_nuclei_image(await nuclei.read(), nuclei.filename)
        img_sec_hash = pipeline.nuclei_hash()
        if await redis.exists(RedisKeys.result_key(img_sec_hash)):
            nucl_cache = await redis.get(RedisKeys.result_key(img_sec_hash))
    else:
        img_sec_hash = None

    # Execute Pipeline
    pipeline._myosam_predictor.update_amg_config(config.amg_config)
    pipeline.set_measure_unit(config.general_config.measure_unit)

    try:
        result = pipeline.execute(myo_cache, nucl_cache).information_metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️ Pipeline has failed: {e}")

    myos, nucls = result.myotubes, result.nucleis
    general_info = preprocess_ws_resp(
        result.model_dump(exclude={"myotubes", "nucleis", "nuclei_clusters"})
    )

    if hasattr(myotube, "filename") and not myo_cache:
        await redis.set(RedisKeys.result_key(img_hash), myos.model_dump_json())
    if hasattr(myotube, "filename") and not nucl_cache:
        await redis.set(RedisKeys.result_key(img_sec_hash), nucls.model_dump_json())

    # Overlay contours on main image
    path = get_fp(STATIC_IMAGES_DIR)
    img = pipeline.draw_contours_on_myotube_image(myotubes=myos, nucleis=nucls)
    pipeline.save_image(path, img)

    return InferenceResponse(
        image_path=path,
        image_hash=img_hash,
        image_secondary_hash=img_sec_hash,
        general_info=general_info,
    )


@router.websocket("/{hash_str}/{sec_hash_str}")
async def inference_ws(
    websocket: WebSocket,
    hash_str: str,
    sec_hash_str: Union[str, None],
    redis: aioredis.Redis = Depends(setup_redis),
) -> None:
    """Websocket for inference mode."""
    if hash_str and hash_str != "null":
        myotubes = Myotubes.model_validate_json(
            await redis.get(RedisKeys.result_key(hash_str))
        )
    else:
        myotubes = Myotubes()

    if sec_hash_str and sec_hash_str != "null":
        nucleis = Nucleis.model_validate_json(
            await redis.get(RedisKeys.result_key(sec_hash_str))
        )
    else:
        nucleis = Nucleis()

    if myotubes and nucleis:
        clusters = NucleiClusters.compute_clusters(nucleis)
    else:
        clusters = NucleiClusters()

    await websocket.accept()
    # Initial websocket sends general information like total number area ...
    # websockets awaits on (x, y) and sends back instance specific information
    while True:
        if len(myotubes) + len(nucleis) == 0:
            await websocket.close()
            break
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            # print("Websocket disconnected.")
            break
        try:
            point = Point.model_validate(data)
        except ValidationError:
            raise WebSocketException(1008, "⚠️ Invalid data received.")
        myo = myotubes.get_instance_by_point(point.to_tuple())

        if myo:
            clusts = clusters.get_clusters_by_myotube_id(myo.identifier)
            resp: dict[str, Any] = {
                "info_data": {
                    "myotube": preprocess_ws_resp(
                        myo.model_dump(), exclude=["roi_coords", "nuclei_ids"]
                    ),
                    "clusters": [clust.model_dump() for clust in clusts],
                }
            }
        else:
            resp = {"info_data": {"myotube": None, "clusters": None}}
        await websocket.send_json(resp)


@router.get("/get_data/{myotube_hash}/{nuclei_hash}")
async def get_data(
    myotube_hash: Union[str, None],
    nuclei_hash: Union[str, None],
    redis: aioredis.Redis = Depends(setup_redis),
) -> dict[str, Any]:
    """Get the data for specific image."""
    if myotube_hash and myotube_hash != "null":
        myotubes = Myotubes.model_validate_json(
            await redis.get(RedisKeys.result_key(myotube_hash))
        )
    else:
        myotubes = Myotubes()

    if nuclei_hash and nuclei_hash != "null":
        nucleis = Nucleis.model_validate_json(
            await redis.get(RedisKeys.result_key(nuclei_hash))
        )
        clusters = NucleiClusters.compute_clusters(nucleis)
    else:
        nucleis = Nucleis()

    if myotubes and nucleis:
        clusters = NucleiClusters.compute_clusters(nucleis)
    else:
        clusters = NucleiClusters()

    return {
        "myotubes": [myo.model_dump(mode="json") for myo in myotubes],
        "nucleis": [nucl.model_dump(mode="json") for nucl in nucleis],
        "nuclei_clusters": [clust.model_dump(mode="json") for clust in clusters],
    }
