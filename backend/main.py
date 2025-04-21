import tempfile
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from myo_sam.inference.models.base import MyoObjects, Myotubes
from myo_sam.inference.pipeline import Pipeline
from read_roi import read_roi_zip
from redis import asyncio as aioredis  # type: ignore

from backend import KEYS, SETTINGS
from backend.dependancies import set_pipeline, setup_redis
from backend.models import Config, State
from backend.routers import inference, validation
from backend.utils import clean_dir

origins = ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan of application."""
    pipeline = set_pipeline(Pipeline())
    # print("> Loading Stardist model...")
    pipeline._stardist_predictor.set_model(SETTINGS.stardist_model)
    # print("> Loading Myosam model...")
    pipeline._myosam_predictor.set_model(SETTINGS.myosam_model, SETTINGS.device)
    yield
    # print("> Cleaning images directory...")
    clean_dir(SETTINGS.images_dir)
    # print("> Done.")


app = FastAPI(lifespan=lifespan, title="MyoVision API", version="0.1.0")
app.include_router(router=inference.router)
app.include_router(validation.router)
app.mount("/images", StaticFiles(directory=SETTINGS.images_dir), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "App developed for the MyoVision project 🚀"}


@app.get("/redis_status/", dependencies=[Depends(setup_redis)])
async def redis_status() -> dict[str, bool]:
    """check status of the redis connection"""
    return {"status": True}


@app.get("/get_config_schema/")
async def get_config_schema() -> dict[str, Any]:
    """Get the configuration of the pipeline."""
    return Config(measure_unit=1.0).model_json_schema()["$defs"]


@app.get("/get_contours/{hash_str}")
async def get_contours(
    hash_str: str, redis: aioredis.Redis = Depends(setup_redis)
) -> dict[str, list[list[list[int]]]]:
    """Get the contours for specific image."""
    objs = await redis.get(KEYS.result_key(hash_str))
    if not objs:
        raise HTTPException(404, "⚠️ Contours not found.")
    objs = MyoObjects.model_validate_json(objs)
    return {"roi_coords": [x.roi_coords for x in objs]}


@app.post("/upload_contours/{hash_str}/")
async def upload_contours(
    hash_str: str,
    redis: aioredis.Redis = Depends(setup_redis),
    file: UploadFile = File(...),
) -> dict[str, list[list[list[int]]]]:
    """Upload the contours of the image."""
    with tempfile.NamedTemporaryFile() as f:
        open(f.name, "wb").write(await file.read())
        try:
            rois_myotubes = read_roi_zip(f.name)
        except Exception:
            raise HTTPException(400, "⚠️ Invalid file format.")
        finally:
            f.close()

    coords_lst = [
        [[x, y] for x, y in zip(roi["x"], roi["y"])] for _, roi in rois_myotubes.items()
    ]

    if not coords_lst:
        raise HTTPException(500, detail="⚠️ Failed to read ROIs.")
    objs_json = await redis.get(KEYS.result_key(hash_str))
    state = State.model_validate_json(await redis.get(KEYS.state_key(hash_str)))
    if not objs_json:
        objects = Myotubes()
    else:
        objects = Myotubes.model_validate_json(objs_json)
    objects.add_instances_from_coords(coords_lst)
    # Shift all indices and update valid set
    state.shift_all(len(coords_lst))
    state.valid.update(range(len(coords_lst)))

    await redis.set(KEYS.result_key(hash_str), objects.model_dump_json())
    await redis.set(KEYS.state_key(hash_str), state.model_dump_json())
    return {"batched_coords": coords_lst}
