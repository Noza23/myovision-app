from typing import Union
from redis import asyncio as aioredis  # type: ignore
from functools import lru_cache

from fastapi import Depends
from fastapi import HTTPException

from myo_sam.inference.pipeline import Pipeline

from .models import Settings, REDIS_KEYS


settings = Settings(_env_file=".env", _env_file_encoding="utf-8")
pipeline: Pipeline = None
redis_keys = REDIS_KEYS(prefix="myovision")


def get_pipeline_instance() -> Pipeline:
    """Each user gets new pipeline instance, where models are shared."""
    return pipeline.model_copy()


async def setup_redis() -> Union[aioredis.Redis, None]:
    """Get a Redis connection."""
    try:
        redis = await aioredis.from_url(
            settings.redis_url, decode_responses=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to Redis: {e}"
        )
    return redis


@lru_cache
def get_redis(
    redis: Union[aioredis.Redis, None] = Depends(setup_redis),
) -> Union[aioredis.Redis, None]:
    """Cache the Redis connection."""
    return redis
