from contextlib import asynccontextmanager
import json
from fastapi import (
    FastAPI,
    HTTPException,
    status,
    UploadFile,
    WebSocket,
    WebSocketException,
)
from fastapi.responses import HTMLResponse
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
            var ws = new WebSocket("ws://localhost:8000/validation/myovision:image:fake_hash");
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


@app.get("/get_config_schema/", response_model=Config)
async def get_config_schema():
    """Get the configuration of the pipeline."""
    return Config(measure_unit=1.0).model_json_schema()["$defs"]
    return Config(measure_unit=1.0)


@app.get("/redis_status/")
async def redis_status():
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
    fake_hash = redis_keys.result_key("fake_hash")
    path = "images/myotube.png"
    return ValidationResponse(image_hash=fake_hash, image_path=path)


@app.websocket("/validation/{hash_str}")
async def validation_ws(websocket: WebSocket, hash_str: str):
    """Websocket for validation mode."""
    print("Websocket connection openning")
    await websocket.accept()
    print(hash_str)
    if hash_str != redis_keys.result_key("fake_hash"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Hash string not found.",
        )
    mo = MyoObjects.model_validate(
        json.load(open("data/info_data.json"))["nucleis"]
    )
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
        i += step
        print("step: ", step)
        print("Sending next contour")
        await websocket.send_json(
            {"roi_coords": mo[i].roi_coords, "contour_id": i}
        )
