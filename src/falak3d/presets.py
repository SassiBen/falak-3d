from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Preset(BaseModel):
    name: str
    title: str
    description: str = ""
    template: str = "{subject}"
    guidance_scale: float = Field(15.0, ge=1, le=30)
    steps: int = Field(64, ge=16, le=128)
    target_faces: int = Field(20000, ge=500)
    size_m: float = Field(1.0, gt=0)

    def prompt(self, subject: str) -> str:
        return self.template.format(subject=subject.strip())


def load_presets(directory: Path) -> dict[str, Preset]:
    presets = {}
    for path in sorted(directory.glob("*.yaml")):
        preset = Preset(**yaml.safe_load(path.read_text(encoding="utf-8")))
        presets[preset.name] = preset
    if not presets:
        raise FileNotFoundError(f"aucun preset dans {directory}")
    return presets
