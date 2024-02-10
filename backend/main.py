from contextlib import asynccontextmanager
import json
import os
import tempfile

from typing import Optional, Any

from fastapi import (
    FastAPI,
    HTTPException,
    status,
    UploadFile,
    WebSocket,
    WebSocketException,
    File,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from .models import (
    Settings,
    REDIS_KEYS,
    Config,
    ValidationResponse,
    InferenceResponse,
    State,
    Point,
)
from .utils import preprocess_ws_resp
from .base import MyoObjects
from .information import InformationMetrics
import random

import read_roi
import numpy as np


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

_ = app.mount("/images", StaticFiles(directory="./images"), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/inference/tt/null");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.get("/get_config_schema/")
async def get_config_schema() -> dict:
    """Get the configuration of the pipeline."""
    return Config(measure_unit=1.0).model_json_schema()["$defs"]


@app.get("/redis_status/")
async def redis_status() -> dict:
    """check status of the redis connection"""
    if random.randint(0, 1):
        return {"status": True}
    else:
        return {"status": False}


@app.post("/validation/", response_model=ValidationResponse)
async def run_validation(image: UploadFile, config: Config):
    """Run the pipeline in validation mode."""
    print("Recived image: ", image.filename)
    print("Recived config: ", config)
    img_names = os.listdir("images/")

    img_names = [
        x for x in img_names if x.endswith(".png") and x != "inference.png"
    ]
    choice = random.choice(img_names)
    path = "images/" + choice
    fake_hash = redis_keys.result_key(choice.split(".")[0])
    return ValidationResponse(image_hash=fake_hash, image_path=path)


@app.post("/inference/", response_model=InferenceResponse)
async def run_inference(
    config: Config,
    image: UploadFile = File(None),
    image_secondary: UploadFile = File(None),
):
    """Run the pipeline in inference mode."""
    if not hasattr(image, "filename") and not hasattr(
        image_secondary, "filename"
    ):
        raise HTTPException(
            status_code=400,
            detail="Either myotube or nuclei image must be provided.",
        )
    print("Recived config: ", config)
    if hasattr(image, "filename"):
        print("Recived myotube: ", image.filename)
        img_hash = redis_keys.result_key("fake_hash")
    else:
        print("No myotube provided")
        img_hash = None
    if hasattr(image_secondary, "filename"):
        print("Recived nuclei: ", image_secondary.filename)
        image_secondary_hash = redis_keys.result_key("fake_sec_hash")
    else:
        print("No nuclei provided")
        image_secondary_hash = None
    path = "images/inference.png"
    info_data = InformationMetrics.model_validate(
        json.load(open("data/fe4_data.json"))
    )
    general_info = preprocess_ws_resp(
        info_data.model_dump(
            exclude={"myotubes", "nucleis", "nuclei_clusters"}
        )
    )

    return InferenceResponse(
        image_path=path,
        image_hash=img_hash,
        image_secondary_hash=image_secondary_hash,
        general_info=general_info,
    )


@app.get("/get_contours/{hash_str}")
async def get_contours(hash_str: str):
    """Get the contours of the image."""
    print("Recived hash: ", hash_str)
    mo = MyoObjects.model_validate(
        json.load(open("data/fe2_data.json"))["myotubes"]
    )
    mo = mo[:4]
    return {"roi_coords": [x.roi_coords for x in mo]}


@app.post("/upload_contours/")
async def upload_contours(file: UploadFile = File(...)):
    """Upload the contours of the image."""
    contents = file.file.read()
    with tempfile.NamedTemporaryFile() as f:
        open(f.name, "wb").write(contents)
        try:
            rois_myotubes = read_roi.read_roi_zip(f.name)
        finally:
            f.close()

    coords_lst = []
    for _, roi in rois_myotubes.items():
        coords = np.round(np.stack((roi["x"], roi["y"]), axis=1)).astype(
            np.int32
        )
        coords_lst.append(coords)
    if not coords_lst:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No contours found.",
        )
    coords = [x.tolist() for x in coords_lst[:4]]
    return {"batched_coords": coords}


@app.websocket("/inference/{hash_str}/{sec_hash_str}")
async def inference_ws(
    websocket: WebSocket, hash_str: Optional[str], sec_hash_str: Optional[str]
) -> None:
    """Websocket for inference mode."""
    if not hash_str and not sec_hash_str:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Hash string missing.",
        )

    print("Websocket connection for inference openning")
    print("Primary hash: ", hash_str)
    print("Secondary hash: ", sec_hash_str)
    await websocket.accept()
    info_data = InformationMetrics.model_validate(
        json.load(open("data/fe4_data.json"))
    )
    myotubes = info_data.myotubes
    nuclei_clusters = info_data.nuclei_clusters

    # Initial websocket sends general information like total number area ...
    # websockets awaits on (x, y) and sends back instance specific information
    while True:
        print("Entered")

        websocket.send_json(
            {
                "info_data": info_data.model_dump(
                    exclude={"myotubes", "nucleis", "nuclei_clusters"}
                )
            }
        )
        if len(info_data.myotubes) + len(info_data.nucleis) == 0:
            print("No data to send")
            websocket.close()
            break

        # Wating for response from front {x: int, y: int}
        data = await websocket.receive_json()

        try:
            point = Point.model_validate(data)
            print("Recived data: ", "point: ", point)
        except ValueError:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Invalid data received.",
            )
        myo = myotubes.get_instance_by_point(point.to_tuple())
        if myo:
            clusts = nuclei_clusters.get_clusters_by_myotube_id(myo.identifier)
            resp: dict[str, Any] = {
                "info_data": {
                    "myotube": preprocess_ws_resp(myo.model_dump()),
                    "clusters": [clust.model_dump() for clust in clusts],
                }
            }
        else:
            resp = {"info_data": {"myotube": None, "clusters": None}}

        await websocket.send_json(resp)


@app.websocket("/validation/{hash_str}")
async def validation_ws(websocket: WebSocket, hash_str: str) -> None:
    """Websocket for validation mode."""
    print("Websocket connection openning")
    await websocket.accept()
    print(hash_str)
    mo_nm = hash_str.split(":")[-1][:3]
    mo = MyoObjects.model_validate(
        json.load(open(f"data/{mo_nm}_data.json"))["myotubes"]
    )
    state = State()

    if state.done:
        await websocket.close()
        return
    i = state.get_next()
    # Starting Contour send on connection openning
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
            await websocket.send_json(
                {
                    "roi_coords": None,
                    "contour_id": i + 1,
                    "total": len(mo),
                }
            )
            await websocket.close()
            break
        else:
            # Wating for response from front
            data = int(await websocket.receive_text())
            # Invalid = 0, Valid = 1, Skip = 2, Undo = -1
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
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Invalid data received.",
                )
            # Send next contour
            step = data != 2 if data != -1 else -1
            i = min(max(i + step, 0), len(mo) - 1)
            print("step: ", step)
            print("Sending next contour")
            await websocket.send_json(
                {
                    "roi_coords": mo[i].roi_coords,
                    "contour_id": i + 1,
                    "total": len(mo),
                }
            )
