import json
from collections import OrderedDict
from typing import Literal, NamedTuple, Self

from myosam.inference.predictors.config import AmgConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypeAlias


class HealthCheck(BaseModel):
    """Health check response model."""

    status: str = "OK"


class RootInfo(BaseModel):
    """Root endpoint response model."""

    message: str = "Rest API for MyoVision application"


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

    model_config = ConfigDict(use_attribute_docstrings=True)

    measure_unit: float = Field(default=1.0, ge=0.0)
    """Measure unit for the uploaded image."""
    invert_image: bool = False
    """Whether to invert the image (for visualization only)."""


class Config(BaseModel):
    """Configuration for the pipeline."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    amg_config: AmgConfig = Field(default_factory=AmgConfig, title="AMG Config")
    """Config for AMG (Auto Mask Generation algorithm)."""
    general_config: GeneralConfig = Field(
        default_factory=GeneralConfig, title="General Config"
    )
    """Config for general pipeline settings."""

    @model_validator(mode="before")
    @classmethod
    def validate_from_json(cls, value):
        """Validate if the input is a JSON string and parse it."""
        if isinstance(value, str):
            return cls.model_validate_json(value)
        return value

    @property
    def mu(self) -> float:
        """Get the measure unit from the general config."""
        return self.general_config.measure_unit

    @property
    def invert(self) -> bool:
        """Get the invert image flag from the general config."""
        return self.general_config.invert_image


# NOTE: ValidationStatus: -1: Invalid, 0: No decision, 1: Valid
ValidationStatus: TypeAlias = Literal[-1, 0, 1]


class State(OrderedDict[int, ValidationStatus]):
    """State of the validation process which is an OrderedDict with specific structure.

    ```
    ValidationStatus = Literal[-1, 0, 1]
    {id: ValidationStatus}
    ```
    - `ValidationStatus`: -1: Invalid, 0: No decision, 1: Valid
    - `id`: list index for the contour in the image with matching `image_id`.
    """

    @classmethod
    def from_dict(cls, dd: dict) -> Self:
        """Create a State object from a dictionary."""
        return cls({int(k): v for k, v in dd.items()})

    @classmethod
    def from_json(cls, s: str) -> Self:
        """Load the state from a JSON string."""
        return json.loads(s=s, object_hook=cls.from_dict)

    def to_json(self) -> str:
        """Dump the state to a JSON string."""
        return json.dumps(self)

    @property
    def done(self) -> bool:
        """Check if the validation is done."""
        return all(v != 0 for v in self.values())

    @property
    def valid(self) -> list[int]:
        """Get a list of all valid contour indices."""
        return [i for i, v in self.items() if v == 1]

    @property
    def invalid(self) -> list[int]:
        """Get a list of all invalid contour indices."""
        return [i for i, v in self.items() if v == -1]

    @property
    def no_decision(self) -> list[int]:
        """Get a list of all contour indices with no decision."""
        return [i for i, v in self.items() if v == 0]

    def undo(self, index: int) -> None:
        """Undo the last decision for a contour."""
        self[index] = 0  # NOTE: Reset the decision to 'No decision'

    def mark_valid(self, index: int) -> None:
        """Mark a contour as valid."""
        self[index] = 1  # NOTE: Mark as valid

    def mark_invalid(self, index: int) -> None:
        """Mark a contour as invalid."""
        self[index] = -1  # NOTE: Mark as invalid

    def skip(self, index: int) -> None:
        """Skip a contour (move it to the end of the queue)."""
        self.move_to_end(index)

    def next(self) -> int | None:
        """Get the next contour index to validate or None if all are validated."""
        for i, v in self.items():
            if v == 0:
                return i
        return None  # NOTE: No contours left to validate

    def reset(self) -> None:
        """Reset the validation state."""
        for k in self.keys():
            self[k] = 0  # NOTE: Reset all decisions to 'No decision'

    def add_valids(self, n: int) -> None:
        """Add a number of valid contours at the beginning of the state."""
        # NOTE: Create a new state with n valid contours
        state = State(dict.fromkeys(range(n), 1))
        # NOTE: Shift all indices by n
        state.update(State({k + n: v for k, v in self.items()}))
        self.clear()
        self.update(state)

    def add_no_decisions(self, n: int) -> None:
        """Add a number of contours with no decision at the end of the state."""
        len_ = len(self)
        self.update(dict.fromkeys(range(len_, len_ + n), 0))


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


class Point(NamedTuple):
    """A point in the 2D space."""

    x: int
    y: int
