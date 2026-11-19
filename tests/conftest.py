from pathlib import Path

import pytest

from falak3d.config import Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(api_key="test-key", data_dir=tmp_path, presets_dir=ROOT / "presets")
