from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    """Configuration for the Redis server."""

    host: str
    """Hostname for the Redis server."""
    port: int = 6379
    """Port for the Redis server."""


class Settings(BaseSettings):
    """Configuraion for the MyoVision Rest API."""

    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=".env")

    redis: RedisSettings
    """Redis server configuration."""
    myosam_model: str
    """Path to the MyoSam model."""
    stardist_model: str
    """Path to the StarDist model."""
    device: str
    """Device to use for computation (e.g., 'cpu', 'cuda')."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application singleton settings."""
    return Settings()
