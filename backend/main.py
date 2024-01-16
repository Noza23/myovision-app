from contextlib import asynccontextmanager
import json
from typing import Annotated

from fastapi import (
    FastAPI,
    HTTPException,
    status,
    UploadFile,
    Depends,
    WebSocket,
    WebSocketException,
)
from fastapi.middleware.cors import CORSMiddleware

from .models import Settings, REDIS_KEYS, Config, ValidationResponse, State
from .base import MyoObjects
import random

settings = Settings(_env_file=".env", _env_file_encoding="utf-8")
# pipeline: Pipeline = None
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
    # global pipeline
    # pipeline = Pipeline()
    # pipeline._stardist_predictor.set_model(settings.stardist_model)
    # pipeline._myosam_predictor.set_model(settings.myosam_model)
    yield
    # Clean up and closing redis
    # del pipeline
    # await redis.close()
    print("Stopping application")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get_config/", response_model=Config)
async def get_config():
    """Get the configuration of the pipeline."""
    return Config(measure_unit=1.0)


@app.get("/redis_status/")
async def redis_status():
    """check status of the redis connection"""
    if random.randint(0, 1):
        return {"status": True}
    else:
        return {"status": False}


@app.post("/validation/", response_model=ValidationResponse)
async def run_validation(
    image: UploadFile,
    config: Config,
    keys: Annotated[REDIS_KEYS, Depends(REDIS_KEYS)],
):
    """Run the pipeline in validation mode."""
    print("Recived image: ", image.filename)
    print("Recived config: ", config)
    fake_hash = keys.result_key("fake_hash")
    path = "images/myotube.png"
    return ValidationResponse(hash_str=fake_hash, image_path=path)


@app.websocket("/validation/{hash_str}/")
async def validation_ws(
    websocket: WebSocket,
    hash_str: str,
    keys: Annotated[REDIS_KEYS, Depends(REDIS_KEYS)],
):
    """Websocket for validation mode."""
    await websocket.accept()
    if hash_str != keys.result_key("fake_hash"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Hash string not found.",
        )
    mo = MyoObjects.model_validate(json.load(open("data/info_data.json")))
    state = State()

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
            print("Invalid contour")
            state.invalid.add(i)
        elif data == 1:
            print("Valid contour")
            state.valid.add(i)
        elif data == 2:
            print("Skip contour")
            _ = mo.move_object_to_end(i)
        elif data == -1:
            print("Undo contour")
            state.valid.discard(i)
            state.invalid.discard(i)
        else:
            raise WebSocketException(
                status_code=status.WS_1008_POLICY_VIOLATION,
                detail="Invalid data received.",
            )
        # Send next contour
        step = data != 2 if data != -1 else -1
        print("step: ", step)
        print("Sending next contour")
        await websocket.send_json(
            {"roi_coords": mo[i + step].roi_coords, "contour_id": i + step}
        )
