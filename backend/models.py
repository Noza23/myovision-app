from pydantic_settings import BaseSettings
from pydantic import BaseModel, model_validator, Field

from enum import Enum
from typing import Union
import json

from myo_sam.inference.models.information import InformationMetrics
from myo_sam.inference.predictors.config import AmgConfig


class Config(BaseModel):
    """Configuration for the pipeline."""

    amg_config: AmgConfig
    measure_unit: float = Field(description="The measure unit in pixels.")

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


class ENDPOINTS(str, Enum):
    VALIDATION = "validation"
    INFERENCE = "inference"


class ValidationResponse(BaseModel):
    """Validation response."""

    roi_coords: list[list[int]] = Field(description="Contours of the ROI.")
    state: dict[str, list[int]] = Field(description="validation state.")
    hash_str: str = Field(description="The hash string of the image.")


class InferenceResponse(BaseModel):
    """Inference response."""

    information_data: InformationMetrics
    hash_str_myotube: Union[str, None] = Field(
        description="The hash string of the myotube image."
    )
    hash_str_nuclei: str[str, None] = Field(
        description="The hash string of the nuclei image."
    )


class REDIS_KEYS:
    """Methods to generate key names for Redis data."""

    def __init__(self, prefix: str = "myovision"):
        self.prefix = prefix

    def image_hash_key(self, hash_str: str) -> str:
        """A key for image hash."""
        return f"{self.prefix}:image:{hash_str}"

    def image_path_key(self, hash_str: str) -> str:
        """A key for image path."""
        return f"{self.prefix}:image_path:{hash_str}"

    def state_key(self, hash_str: str) -> str:
        """A key for state."""
        return f"{self.prefix}:state:{hash_str}"
