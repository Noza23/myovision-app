from myosam.inference.predictors.config import AmgConfig
from pydantic import BaseModel, Field, model_validator


class HealthCheck(BaseModel):
    """Health check response model."""

    status: str = "OK"


class RootInfo(BaseModel):
    """Root endpoint response model."""

    message: str = "App developed for the MyoVision project 🚀"


class Contour(BaseModel):
    """Single contour in an image."""

    coords: list[list[int]]
    """Coordinates in N-dim space, where each point is represented by list(n)"""


class ImageContours(BaseModel):
    """Response model for all contours in an image."""

    contours: list[Contour]
    """List of all contours in an image."""


class GeneralConfig(BaseModel):
    """General configuration for the pipeline."""

    measure_unit: float = Field(
        default=1.0,
        ge=0.0,
        description="The measure unit for the uploaded image.",
    )
    invert_image: bool = Field(
        default=False,
        description="Whether to invert the image (for visualization only).",
    )


class Config(BaseModel):
    """Configuration for the pipeline."""

    amg_config: AmgConfig = Field(
        default_factory=AmgConfig,
        description="Config for AMG (Auto Mask Generation) algorithm.",
        title="AMG Config",
    )
    general_config: GeneralConfig = Field(
        default_factory=GeneralConfig,
        description="General configuration for the pipeline.",
        title="General Config",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_from_json(cls, value):
        """Validate if the input is a JSON string and parse it."""
        if isinstance(value, str):
            return cls.model_validate_json(value)
        return value


class State(BaseModel):
    """Validation state."""

    valid: set[int] = Field(default_factory=set)
    """Set of indicies of valid contours."""
    invalid: set[int] = Field(default_factory=set)
    """Set of indicies of invalid contours."""
    done: bool = Field(default=False)
    """Wtether the validation is done or not."""

    def get_next(self) -> int:
        """Get the next contour index."""
        combined = self.valid | self.invalid
        if len(combined) == 0:
            return 0
        return max(combined) + 1

    def shift_positive(self, shift: int) -> None:
        """Shift all indices to the right, assuming all indices are positive."""
        self.valid = {i + shift for i in self.valid}
        self.invalid = {i + shift for i in self.invalid}
        return self.valid.update(range(shift))


class ValidationResponse(BaseModel):
    """Response model for validation endpoint."""

    image_hash: str
    """Image Hash, that is treated as a unique identifier across the application."""
    image_path: str
    """Image path in the static cache directory."""


class InferenceResponse(BaseModel):
    """Inference response."""

    image_path: str
    """Path to the image with contours overlayed."""
    image_hash: str | None
    """Image hash, that is treated as a unique identifier across the application."""
    image_secondary_hash: str | None
    """Secondary image hash, that is treated as a unique identifier across the application."""
    general_info: dict[str, str | float]
    """General information about the inference."""


class Point(BaseModel):
    """A point on the image."""

    x: int = Field(ge=0)
    """x coordinate of the point."""
    y: int = Field(ge=0)
    """y coordinate of the point."""

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple."""
        return (self.x, self.y)
