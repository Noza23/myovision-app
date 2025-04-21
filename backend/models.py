from typing import Union

from myosam.inference.predictors.config import AmgConfig
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


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
    def validate_from_json(cls, value):
        if isinstance(value, str):
            return cls.model_validate_json(value)
        return value


class Settings(BaseSettings):
    """App confiuration."""

    redis_url: str = Field(description="Redis URL.")
    myosam_model: str = Field(description="MyoSam model path.")
    stardist_model: str = Field(description="StarDist model path.")
    images_dir: str = Field(description="Images directory.")
    backend_port: int = Field(description="Backend port.", ge=0)
    frontend_port: int = Field(description="Frontend port.", ge=0)
    web_concurrency: int = Field(description="Web concurrency.", ge=0)
    device: str = Field(description="Device to use for computation.")


class State(BaseModel):
    """Validation state."""

    valid: set[int] = Field(description="valid contours.", default_factory=set)
    invalid: set[int] = Field(description="invalid contours.", default_factory=set)
    done: bool = Field(description="validation done.", default=False)

    def get_next(self) -> int:
        """Get the next contour index."""
        combined = self.valid | self.invalid
        if len(combined) == 0:
            return 0
        return max(combined) + 1

    def shift_all(self, shift: int) -> None:
        """Shift all indices."""
        self.valid = {i + shift for i in self.valid}
        self.invalid = {i + shift for i in self.invalid}


class ValidationResponse(BaseModel):
    """Validation response."""

    image_hash: str = Field(description="The hash string of the image.")
    image_path: str = Field(description="The path of the image.")


class InferenceResponse(BaseModel):
    """Inference response."""

    image_path: str = Field(description="The path of the image.")
    image_hash: Union[str, None] = Field(description="The hash string of the image.")
    image_secondary_hash: Union[str, None] = Field(
        description="The hash string of the secondary image."
    )
    general_info: dict[str, Union[str, float]] = Field(
        description="General information about segmentation"
    )


class Point(BaseModel):
    """A point on the image"""

    x: int = Field(description="x coordinate of the point", ge=0)
    y: int = Field(description="y coordinate of the point", ge=0)

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple."""
        return (self.x, self.y)
