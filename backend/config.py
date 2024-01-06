from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App confiuration."""

    redis_url: str
    MYOSAM_MODEL: str
    STARDIST_MODEL: str
    measure_unit: int
