import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    data_dir: Path = field(default_factory=lambda: Path("data"))
    presets_dir: Path = field(default_factory=lambda: Path("presets"))
    max_queue: int = 10
    retention_days: int = 7
    ollama_url: str | None = None
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_S"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("FALAK3D_API_KEY", ""),
            data_dir=Path(os.getenv("FALAK3D_DATA_DIR", "data")),
            presets_dir=Path(os.getenv("FALAK3D_PRESETS_DIR", "presets")),
            max_queue=int(os.getenv("FALAK3D_MAX_QUEUE", "10")),
            retention_days=int(os.getenv("FALAK3D_RETENTION_DAYS", "7")),
            ollama_url=os.getenv("OLLAMA_URL") or None,
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
        )
