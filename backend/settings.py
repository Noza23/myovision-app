import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    """Configuration for the Redis server."""

    host: str
    """Hostname for the Redis server."""
    port: int = 6379
    """Port for the Redis server."""


class Settings(BaseSettings):
    """Configuraion for the MyoVision Rest API."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        extra="allow",
        env_prefix="MYOVISION_",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """Logging level for the application."""
    redis: RedisSettings
    """Redis server configuration."""
    myosam_model: str
    """Path to the MyoSam model."""
    stardist_model: str
    """Path to the StarDist model."""
    device: str = "cpu"
    """Device to use for computation (e.g., 'cpu', 'cuda')."""
    cache_dir: str = "static/images"
    """Directory for caching images."""

    @field_validator("cache_dir")
    def validate_cache_dir(cls, v):
        """Ensure cache directory exists."""
        if not Path(v).exists():
            os.makedirs(v, exist_ok=True)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application singleton settings."""
    return Settings()
