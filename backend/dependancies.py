from typing import Union
from redis import asyncio as aioredis  # type: ignore
from functools import lru_cache

from fastapi import HTTPException, Depends

from myo_sam.inference.pipeline import Pipeline

from backend import SETTINGS

pipeline: Pipeline = None


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
    return redis


@lru_cache
def get_redis(redis: aioredis.Redis = Depends(setup_redis)) -> aioredis.Redis:
    """Cache the Redis connection."""
    return redis


async def get_status(redis: aioredis.Redis) -> None:
    """Check if Redis is available."""
    if not await redis.ping():
        raise HTTPException(status_code=500, detail="⚠️ Redis not available.")
