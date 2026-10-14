"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML configuration file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
