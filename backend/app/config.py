from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlserver_url: str = Field(default="http://localhost:8080", alias="MLSERVER_URL")
    models_dir: Path = Field(default=Path("/models"), alias="MODELS_DIR")
    feature_names: list[str] = ["temperature", "pressure", "vibration", "humidity"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
