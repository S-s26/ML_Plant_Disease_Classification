"""Configuration and serialization helpers."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration and resolve project root paths."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(rel_or_abs: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return its Path."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> Path:
    """Save JSON data with readable indentation."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return p


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
