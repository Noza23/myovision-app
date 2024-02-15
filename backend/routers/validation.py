from typing import Union

from fastapi import APIRouter, UploadFile, Depends, WebSocket
from fastapi import HTTPException, status, WebSocketException
from redis import asyncio as aioredis  # type: ignore

from myo_sam.inference.pipeline import Pipeline
from myo_sam.inference.models.base import Myotubes
from myo_sam.inference.predictors.utils import invert_image

from backend import KEYS, SETTINGS
from backend.utils import get_fp
from backend.models import Config, ValidationResponse, State
from backend.dependancies import get_redis, get_pipeline_instance, get_status


router = APIRouter(
    prefix="/validation",
    tags=["validation"],
    dependencies=[Depends(get_status)],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=ValidationResponse)
async def run_validation(
    image: UploadFile,
    config: Config,
    redis: Union[aioredis.Redis, None] = Depends(get_redis),
    pipeline: Pipeline = Depends(get_pipeline_instance),
):
    """Run the pipeline in validation mode."""
    pipeline.set_myotube_image(await image.read(), image.filename)
    img_hash = pipeline.myotube_hash()

    img_to_send = pipeline.myotube_image_np.copy()
    if config.general_config.invert_image:
        img_to_send = invert_image(img_to_send)

    if await redis.exists(KEYS.result_key(img_hash)):
        # Case when image is cached
        myos = Myotubes.model_validate_json(
            await redis.get(KEYS.result_key(img_hash))
        )
        if myos[0].measure_unit != config.general_config.measure_unit:
            myos.adjust_measure_unit(config.general_config.measure_unit)
            await redis.set(KEYS.result_key(img_hash), myos.model_dump_json())
        state = State.model_validate_json(
            await redis.get(KEYS.state_key(img_hash))
        )
        if state.valid:
            img_to_send = pipeline.draw_contours(
                img_to_send,
                [myos[i].roi_coords_np for i in state.valid],
                color=(0, 255, 0),
            )
    else:
        # Case when image is not cached
        state = State()
        pipeline._myosam_predictor.update_amg_config(config.amg_config)
        pipeline.set_measure_unit(config.general_config.measure_unit)
        try:
            myos = pipeline.execute().information_metrics.myotubes
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"⚠️ Pipeline has failed: {e}"
            )

        await redis.mset(
            {
                KEYS.result_key(img_hash): myos.model_dump_json(),
                KEYS.state_key(img_hash): state.model_dump_json(),
            }
        )
    path = get_fp(SETTINGS.images_dir)
    pipeline.save_image(path, img_to_send)
    return ValidationResponse(image_hash=img_hash, image_path=path)


@router.websocket("/{hash_str}")
async def validation_ws(
    websocket: WebSocket,
    hash_str: str,
    redis: Union[aioredis.Redis, None] = Depends(get_redis),
) -> None:
    """Websocket for validation mode."""

    await websocket.accept()
    print("Hash Received: ", hash_str)

    mo = Myotubes.model_validate_json(
        await redis.get(KEYS.result_key(hash_str))
    )
    state = State.model_validate_json(
        await redis.get(KEYS.state_key(hash_str))
    )
    i = state.get_next()

    if state.done:
        websocket.close()
        return

    await websocket.send_json(
        {
            "roi_coords": mo[i].roi_coords,
            "contour_id": i + 1,
            "total": len(mo),
        }
    )

    while True:
        data = int(await websocket.receive_text())
        if data == 0:
            state.invalid.add(i)
        elif data == 1:
            state.valid.add(i)
        elif data == 2:
            _ = mo.move_object_to_end(i)
            await redis.set(KEYS.result_key(hash_str), mo.model_dump_json())
        elif data == -1:
            state.valid.discard(i)
            state.invalid.discard(i)
        else:
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

        await redis.set(KEYS.state_key(hash_str), state.model_dump_json())
        step = data != 2 if data != -1 else -1
        i = min(max(i + step, 0), len(mo) - 1)

        if i == len(mo):
            state.done = True
            await redis.set(KEYS.result_key(hash_str), mo.model_dump_json())
            break

        # Send next contour
        await websocket.send_json(
            {
                "roi_coords": mo[i].roi_coords,
                "contour_id": i + 1,
                "total": len(mo),
            }
        )
