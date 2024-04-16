from fastapi import HTTPException
from redis import asyncio as aioredis  # type: ignore

from myo_sam.inference.pipeline import Pipeline

from backend import SETTINGS

pipeline: Pipeline = None


def set_pipeline(p: Pipeline) -> Pipeline:
    """Set the pipeline instance on startup."""
    global pipeline
    pipeline = p
    return pipeline


def get_pipeline_instance() -> Pipeline:
    """Each user gets new pipeline instance, where models are shared."""
    return pipeline.model_copy()


async def setup_redis() -> aioredis.Redis:
    """Get a Redis connection."""
    try:
        redis = await aioredis.from_url(
            SETTINGS.redis_url, decode_responses=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"⚠️ Failed to connect to Redis: {e}"
        )
    try:
        await redis.ping()
    except Exception:
        raise HTTPException(status_code=500, detail="⚠️ Redis not available.")
    return redis
