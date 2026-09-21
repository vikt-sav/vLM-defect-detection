from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEFECTBENCH_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    categories: str = "bottle,cable"
    vlm_model: str = "qwen2.5vl:3b"
    vlm_timeout: float = 180.0
    ollama_host: str = "http://localhost:11434"

    @property
    def category_list(self) -> list[str]:
        return [c.strip() for c in self.categories.split(",") if c.strip()]

    @property
    def mvtec_root(self) -> Path:
        return self.data_dir / "mvtec_anomaly_detection"


def get_settings() -> Settings:
    return Settings()
