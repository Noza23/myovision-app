import os
import logging
from pathlib import Path

from myosam.inference.pipeline import Pipeline
from redis.asyncio.client import Redis as _Redis

from backend.settings import get_settings

logger = logging.getLogger(__name__)

class MyoRedis(_Redis):
    """Redis client for MyoVision application."""

    prefix: str = "myovision"

    @classmethod
    def key(cls, image_id: str) -> str:
        """A Key for identifying results of the given image."""
        return f"{cls.prefix}:image:{image_id}"

    def state_key(cls, image_id: str) -> str:
        """A Key for identifying validation state of the given image."""
        return f"{cls.prefix}:state:{image_id}"

    async def get_by_id(self, image_id: str) -> str | None:
        """Get value by image id which in most cases is a hash string."""
        return await self.get(name=self.key(image_id))

    async def set_by_id(self, image_id: str, value: str) -> bool | None:
        """Set value by image id which in most cases is a hash string."""
        return await self.set(name=self.key(image_id), value=value)

    async def get_state_by_id(self, image_id: str) -> str | None:
        """Get validation state by image id."""
        return await self.get(name=self.state_key(image_id))

    async def set_state_by_id(self, image_id: str, value: str) -> bool | None:
        """Set validation state by image id."""
        return await self.set(name=self.state_key(image_id), value=value)


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
            raise FileNotFoundError(f"Cache directory {self.cache_dir} does not exist.")
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

    def _cleanup_cache(self) -> None:
        """Cleanup all files in the specified directory."""
        logger.info(f"Cleaning up cache directory: {self.cache_path}")
        return [os.remove(f) for f in self.cache_path.glob("*") if f.is_file()]

    def get_pipeline(self) -> Pipeline:
        """Get the configured pipeline."""
        if not self._isetup:
            raise RuntimeError("Pipeline is not set up. Call setup() first.")
        # Return a copy of the pipeline to avoice collisions for multiple users
        # model_copy does not copy the model weights, as this is not a deep copy
        # and only copies the configuration.
        return self.pipeline.model_copy()


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
