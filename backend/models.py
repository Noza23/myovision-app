from pydantic_settings import BaseSettings
from pydantic import BaseModel, model_validator, Field

from enum import Enum
import json

from myo_sam.inference.predictors.config import AmgConfig


class GeneralConfig(BaseModel):
    """General configuration for the pipeline."""

    measure_unit: float = Field(
        description="The measure unit for the image.", default=1.0
    )


class Config(BaseModel):
    """Configuration for the pipeline."""

    amg_config: AmgConfig = Field(
        description="Config for AMG algorithm.", default=AmgConfig()
    )
    general_config: GeneralConfig = Field(
        description="General Pipeline Config.", default=GeneralConfig()
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


class ENDPOINTS(str, Enum):
    VALIDATION = "validation"
    INFERENCE = "inference"


class ObjectNames(str, Enum):
    MYOTUBES = "myotubes"
    NUCLEIS = "nucleis"


class State(BaseModel):
    """Validation state."""

    valid: list[int] = Field(description="valid contours.", default=[])
    invalid: list[int] = Field(description="invalid contours.", default=[])


class ValidationResponse(BaseModel):
    """Validation response."""

    roi_coords: list[list[list[int]]] = Field(description="ROI coords.")
    state: State = Field(description="validation state.")
    image_hash: str = Field(description="The hash string of the image.")
    image_path: str = Field(description="The path of the image.")


class InferenceResponse(BaseModel):
    """Inference response."""

    # information_data: InformationMetrics # Maybe not needed due to websocket

    image_path: str = Field(description="The path of the image.")
    image_hash: str = Field(description="The hash string of the image.")
    secondary_image_hash: str = Field(
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
