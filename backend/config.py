from pydantic_settings import BaseSettings
from myo_sam.inference.predictors.config import AmgConfig


class Settings(BaseSettings):
    """App confiuration."""

    redis_url: str
    MYOSAM_MODEL: str
    STARDIST_MODEL: str
    amg_config: AmgConfig = AmgConfig()
    measure_unit: int
