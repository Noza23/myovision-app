from pydantic_settings import BaseSettings
from pydantic import BaseModel, model_validator, Field

from enum import Enum
import json


class AmgConfig(BaseModel):
    """The configuration of a AMG framework."""

    points_per_side: int = Field(
        description="Number of points per side to sample.", default=64
    )
    points_per_batch: int = Field(
        description="Number of points to predict per batch", default=64
    )
    pred_iou_thresh: float = Field(
        description="Threshold for predicted IoU", default=0.8
    )
    stability_score_thresh: float = Field(
        description="Threshold for stability score", default=0.92
    )
    stability_score_offset: float = Field(
        description="Offset for stability score", default=1.0
    )
    box_nms_thresh: float = Field(description="NMS threshold", default=0.7)
    crop_n_layers: int = Field(
        description="Number of layers to crop", default=1
    )
    crop_nms_thresh: float = Field(
        description="NMS threshold for cropping", default=0.7
    )
    crop_overlap_ratio: float = Field(
        description="Overlap ratio for cropping", default=512 / 1500
    )
    crop_n_points_downscale_factor: int = Field(
        description="Downscale factor for cropping", default=2
    )
    min_mask_region_area: int = Field(
        description="Minimum area of mask region", default=100
    )


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
    general_config = Field(
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


class REDIS_KEYS:
    """Methods to generate key names for Redis data."""

    def __init__(self, prefix: str = "myovision"):
        self.prefix = prefix

    def myotube_key(self, hash_str: str) -> str:
        """A key for myotube image."""
        return f"{self.prefix}:myotube:{hash_str}"

    def nuclei_key(self, hash_str: str) -> str:
        """A key for myotube mask."""
        return f"{self.prefix}:nuclei:{hash_str}"

    def image_path_key(self, hash_str: str) -> str:
        """A key for image path."""
        return f"{self.prefix}:image_path:{hash_str}"

    def state_key(self, hash_str: str) -> str:
        """A key for state."""
        return f"{self.prefix}:state:{hash_str}"
