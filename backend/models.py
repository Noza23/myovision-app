from pydantic_settings import BaseSettings
from pydantic import BaseModel
from enum import Enum
from myo_sam.inference.predictors.config import AmgConfig


class Config(BaseModel):
    """Configuration for the pipeline."""

    amg_config: AmgConfig
    measure_unit: float


class Settings(BaseSettings):
    """App confiuration."""

    redis_url: str
    myosam_model: str
    stardist_model: str


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

    def state_key(self, hash_str: str) -> str:
        """A key for state."""
        return f"{self.prefix}:state:{hash_str}"
