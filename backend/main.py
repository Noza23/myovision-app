from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, BackgroundTasks, Depends
from redis import asyncio as aioredis  # type: ignore

# redis.set() caching results ex=1h for given time.


def setup_redis(url: str = "redis://localhost") -> aioredis.Redis:
    # Load models, connect to database, etc.
    try:
        redis = aioredis.from_url(url, decode_responses=True)
        yield redis
    except Exception as e:
        print(f"Failed establishing connection to redis: {e}")
    finally:
        # Clean up
        print("Stopping application")
        redis.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application")
    # Load models
    yield
    # Clean up
    print("Stopping application")


app = FastAPI(lifespan=lifespan)


@app.get("/run")
async def run(
    data: dict,
    background_tasks: BackgroundTasks,
    redis: Annotated[aioredis.Redis, Depends(setup_redis)],
):
    # Run the pipeline

    # cache results in redis

    return {"message": "Hello World"}


async def set_cache(
    data: dict, redis: Annotated[aioredis.Redis, Depends(setup_redis)]
) -> None:
    # Set cache in redis for 1 hour
    await redis.set("key", "value", ex=3600)
