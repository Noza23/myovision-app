from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import Settings, REDIS_KEYS, Config, InferenceResponse
import json
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
async def status():
    """check status of the redis connection"""
    if random.randint(0, 1):
        return {"status": True}
    else:
        return {"status": False}


@app.post("/inference/", response_model=InferenceResponse)
async def run_inference(
    config: Config,
    myotube: UploadFile = File(None),
    nuclei: UploadFile = File(None),
):
    """Run the pipeline in inference mode."""
    if not myotube.filename and not nuclei.filename:
        raise HTTPException(
            status_code=400,
            detail="Either myotube or nuclei image must be provided.",
        )

    # myo_cache, nucl_cache = None, None
    # if myotube.filename:
    #     pipeline.set_myotube_image(await myotube.read(), myotube.filename)
    #     img_hash = pipeline.myotube_hash
    #     if await is_cached(keys.result_key(img_hash), redis):
    #         myo_cache = await redis.get(keys.result_key(img_hash))
    #         path = await redis.get(keys.image_path_key(img_hash))
    #         if not path:
    #             # path might be cleaned by regular image cleaning
    #             path = get_fp(settings.images_dir)
    #             _ = pipeline.save_myotube_image(path)
    #             background_tasks.add_task(
    #                 set_cache, {keys.image_path_key(img_hash): path}, redis
    #             )
    # if nuclei.filename:
    #     pipeline.set_nuclei_image(await nuclei.read(), nuclei.filename)
    #     sec_img_hash = pipeline.nuclei_hash
    #     if await is_cached(keys.result_key(sec_img_hash), redis):
    #         nucl_cache = await redis.get(keys.result_key(sec_img_hash))

    # # Execute Pipeline
    # pipeline._myosam_predictor.update_amg_config(config.amg_config)
    # pipeline.set_measure_unit(config.general_config.measure_unit)
    # result = pipeline.execute(myo_cache, nucl_cache).information_metrics
    # myos, nucls = result.myotubes, result.nucleis

    # if myotube.filename and not myo_cache:
    #     background_tasks.add_task(
    #         set_cache,
    #         {keys.result_key(sec_img_hash): myos.model_dump_json()},
    #     )

    # if nuclei.filename and not nucl_cache:
    #     background_tasks.add_task(
    #         set_cache,
    #         {keys.result_key(sec_img_hash): nucls.model_dump_json()},
    #     )

    # # Overlay contours on main image
    # img_drawn = pipeline.draw_contours_on_myotube_image(myos, nucls)
    # path = get_fp(settings.images_dir)
    # pipeline.save_myotube_image(path, img_drawn)
    # return InferenceResponse(
    #     iamge_path=path, image_hash=img_hash, secondary_image_hash=sec_img_hash
    # )
