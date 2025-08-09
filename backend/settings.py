from functools import lru_cache
from typing import Literal

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
        env_nested_delimiter="__", env_file=".env", extra="allow"
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """Logging level for the application."""
    redis: RedisSettings
    """Redis server configuration."""
    myosam_model: str
    """Path to the MyoSam model."""
    stardist_model: str
    """Path to the StarDist model."""
    device: str
    """Device to use for computation (e.g., 'cpu', 'cuda')."""
    cache_dir: str = "static/images"
    """Directory for caching images."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application singleton settings."""
    return Settings()
