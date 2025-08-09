import json
import logging
import os
from pathlib import Path
from typing import TypeAlias, TypeVar
from uuid import uuid4

from myosam.inference.models.base import MyoObjects, Myotubes, Nucleis
from myosam.inference.pipeline import Pipeline
from pydantic import BaseModel
from redis.asyncio.client import Redis as _Redis

from backend.models import Config, State
from backend.settings import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", BaseModel, State)
Value: TypeAlias = str | BaseModel | State


class MyoRedis(_Redis):
    """Redis client for MyoVision application."""

    prefix = "myovision"

    @classmethod
    def objects_key(cls, image_id: str) -> str:
        """A Key for identifying objects for the given image ID."""
        return f"{cls.prefix}:objects:{image_id}"

    @classmethod
    def state_key(cls, image_id: str) -> str:
        """A Key for identifying validation state for the given image ID."""
        return f"{cls.prefix}:state:{image_id}"

    @staticmethod
    def _serialize_value(value: Value) -> str:
        """Serialize Value type to a string representation for Redis storage."""
        if isinstance(value, str):
            return value
        elif isinstance(value, BaseModel):
            return value.model_dump_json()
        elif isinstance(value, State):
            return value.to_json()
        return json.dumps(value)  # NOTE: Fallback

    @staticmethod
    def _deserialize_value(value: str, model: type[T]) -> T:
        """Deserialize a string value to the specified model type."""
        if issubclass(model, State):
            return model.from_json(value)
        elif issubclass(model, BaseModel):
            return model.model_validate_json(value)
        msg = f"Cannot deserialize value to {model.__name__}. "
        raise TypeError(msg)

    async def get_by_key(self, key: str, model: type[T]) -> T | None:
        """Get value by key and deserialize it to the given model."""
        logger.info(f"[GET] {model.__name__} for key: {key}")
        if not (value := await self.get(key)):
            logger.info(f"[NOT_FOUND] {model.__name__} for key: {key}")
            return None
        return self._deserialize_value(value, model)

    async def set_by_key(self, key: str, value: Value) -> bool | None:
        """Set value by key (serializing it if needed)."""
        logger.info(f"[SET] {type(value).__name__} for key: {key}")
        return await self.set(key, value=self._serialize_value(value))

    async def mset_by_key(self, mapping: dict[str, Value]) -> bool | None:
        """Set multiple values by keys (serializing them if needed)."""
        return await self.mset(
            {
                k: (
                    logger.info(  # type: ignore[func-returns-value]
                        f"[MSET] {type(v).__name__} for key: {k}"
                    )
                    or self._serialize_value(v)
                )
                for k, v in mapping.items()
            }
        )

    async def mget_by_key(self, mapping: dict[str, type[T]]) -> dict[str, T | None]:
        """Get multiple values by keys and deserialize them to the given model."""
        logger.info(
            f"[MGET] {', '.join(f'{v.__name__} for key: {k}' for k, v in mapping.items())}"
        )
        result = dict.fromkeys(mapping.keys(), None)
        # NOTE: MGET returns the values in same order as keys provided
        objects_raw: list[str | None] = await self.mget(mapping.keys())
        for id_, obj_raw in zip(mapping.keys(), objects_raw, strict=True):
            if obj_raw is not None:
                result[id_] = self._deserialize_value(obj_raw, mapping[id_])
            else:
                logger.info(f"[NOT_FOUND] {mapping[id_].__name__} for key: {id_}")
        return result

    async def get_state_by_id(self, image_id: str) -> State | None:
        """Get validation state by image ID."""
        return await self.get_by_key(self.state_key(image_id), model=State)

    async def get_myoobjects_by_id(self, image_id: str) -> MyoObjects | None:
        """Get General MyoObjects by image ID (useful when we only need contours)."""
        return await self.get_by_key(self.objects_key(image_id), model=MyoObjects)

    async def get_myotubes_by_id(self, image_id: str) -> Myotubes | None:
        """Get Myotubes by image ID."""
        return await self.get_by_key(self.objects_key(image_id), model=Myotubes)

    async def get_nucleis_by_id(self, image_id: str) -> Nucleis | None:
        """Get Nucleis by image ID."""
        return await self.get_by_key(self.objects_key(image_id), model=Nucleis)

    async def set_objects_by_id(self, image_id: str, value: Value) -> bool | None:
        """Set MyoObjects by image ID."""
        return await self.set_by_key(self.objects_key(image_id), value=value)

    async def set_state_by_id(self, image_id: str, value: Value) -> bool | None:
        """Set validation State by image ID."""
        return await self.set_by_key(self.state_key(image_id), value=value)

    async def set_objects_and_state_by_id(
        self, image_id: str, objects: Value, state: Value
    ) -> bool | None:
        """Set both MyoObjects and State by image ID."""
        return await self.mset_by_key(
            {self.objects_key(image_id): objects, self.state_key(image_id): state}
        )

    async def get_myotubes_and_nucleis_by_id(
        self, myotubes_id: str, nucleis_id: str
    ) -> tuple[Myotubes | None, Nucleis | None]:
        """Get Myotubes and Nucleis by their IDs."""
        myotubes_key = self.objects_key(myotubes_id)
        nucleis_key = self.objects_key(nucleis_id)
        mapping = {myotubes_key: Myotubes, nucleis_key: Nucleis}
        result = await self.mget_by_key(mapping)
        return (result[myotubes_key], result[nucleis_key])


class MyoSamManager:
    """Wrapper for MyoSam inference pipeline for convenient access and pipeline management."""

    def __init__(
        self, stardist_model: str, myosam_model: str, device: str, cache_dir: str
    ):
        """Initialize the MyoSam pipeline with optional model paths and device."""
        self.stardist_model = stardist_model
        self.myosam_model = myosam_model
        self.device = device
        self.cache_dir = cache_dir
        self.cache_path = Path(self.cache_dir)
        if not self.cache_path.exists():
            msg = f"Cache directory {self.cache_dir} does not exist."
            raise FileNotFoundError(msg)
        self.pipeline = Pipeline()
        self._isetup = False

    def __repr__(self):
        """String representation of the MyoSamManager."""
        return (
            f"MyoSamManager(stardist_model={self.stardist_model}, "
            f"myosam_model={self.myosam_model}, device={self.device})"
        )

    def setup(self):
        """Setup the pipeline with the MyoSam and StarDist models."""
        self.pipeline._stardist_predictor.set_model(self.stardist_model)
        self.pipeline._myosam_predictor.set_model(
            checkpoint=self.myosam_model, device=self.device
        )
        self._isetup = True

    def cleanup(self):
        """Clean up resources."""
        self.pipeline.clear_cache()
        self.pipeline._stardist_predictor.model = None
        self.pipeline._myosam_predictor.model = None
        self._cleanup_cache()
        self._isetup = False

    def generate_fp(self, suffix: str = ".png") -> str:
        """Generate a unique file path in the cache directory."""
        return os.path.join(self.cache_dir, f"{uuid4().hex}{suffix}")

    def _cleanup_cache(self) -> None:
        """Cleanup all files in the specified directory."""
        logger.info(f"Cleaning up cache directory: {self.cache_path}")
        for f in self.cache_path.glob("*"):
            if f.is_file():
                logger.info(f"Removing file: {f}")
                f.unlink()

    def get_pipeline(self, config: Config) -> Pipeline:
        """Get the configured pipeline."""
        if not self._isetup:
            msg = "Pipeline is not set up. Call setup() method first."
            raise RuntimeError(msg)
        # Return a copy of the pipeline to avoice collisions for multiple users
        # model_copy does not copy the model weights, as this is not a deep copy
        # and only copies the configuration.
        pipeline = self.pipeline.model_copy()
        pipeline._myosam_predictor.update_amg_config(config.amg_config)
        pipeline.set_measure_unit(mu=config.mu)
        return pipeline


Redis = MyoRedis(
    host=get_settings().redis.host,
    port=get_settings().redis.port,
    decode_responses=True,
)

MyoSam = MyoSamManager(
    stardist_model=get_settings().stardist_model,
    myosam_model=get_settings().myosam_model,
    device=get_settings().device,
    cache_dir=get_settings().cache_dir,
)
