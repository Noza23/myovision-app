from typing import Union
import json

from pydantic_settings import BaseSettings
from pydantic import BaseModel, model_validator, Field

from myo_sam.inference.predictors.config import AmgConfig


class GeneralConfig(BaseModel):
    """General configuration for the pipeline."""

    measure_unit: float = Field(
        description="The measure unit for the image", default=1.0, ge=0.0
    )
    invert_image: bool = Field(
        description="Whether to invert the image (for visualization only)",
        default=False,
    )


class Config(BaseModel):
    """Configuration for the pipeline."""

    amg_config: AmgConfig = Field(
        description="Config for AMG algorithm.",
        default_factory=AmgConfig,
        title="AMG Config",
    )
    general_config: GeneralConfig = Field(
        description="General Pipeline Config.",
        default_factory=GeneralConfig,
        title="General Config",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class Settings(BaseSettings):
    """App confiuration."""

    redis_url: str
    myosam_model: str
    stardist_model: str
    images_dir: str


class State(BaseModel):
    """Validation state."""

    valid: set[int] = Field(description="valid contours.", default_factory=set)
    invalid: set[int] = Field(
        description="invalid contours.", default_factory=set
    )
    done: bool = Field(description="validation done.", default=False)

    def get_next(self) -> int:
        """Get the next contour index."""
        combined = self.valid | self.invalid
        if len(combined) == 0:
            return 0
        return max(combined) + 1


class ValidationResponse(BaseModel):
    """Validation response."""

    image_hash: str = Field(description="The hash string of the image.")
    image_path: str = Field(description="The path of the image.")


class InferenceResponse(BaseModel):
    """Inference response."""

    image_path: str = Field(description="The path of the image.")
    image_hash: Union[str, None] = Field(
        description="The hash string of the image."
    )
    image_secondary_hash: Union[str, None] = Field(
        description="The hash string of the secondary image."
    )


class REDIS_KEYS:
    """Methods to generate key names for Redis data."""

    def __init__(self, prefix: str = "myovision"):
        self.prefix = prefix

    def result_key(self, hash_str: str) -> str:
        """A key for image hash."""
        return f"{self.prefix}:image:{hash_str}"

    def image_path_key(self, hash_str: str) -> str:
        """A key for image path."""
        return f"{self.prefix}:image_path:{hash_str}"

    def state_key(self, hash_str: str) -> str:
        """A key for state."""
        return f"{self.prefix}:state:{hash_str}"


class Point(BaseModel):
    """A point on the image"""

    x: int = Field(description="x coordinate of the point", ge=0)
    y: int = Field(description="y coordinate of the point", ge=0)

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple."""
        return (self.x, self.y)
