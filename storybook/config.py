"""Carga de configuración (config.yaml) con valores por defecto y presets."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def _deep_update(base: dict, other: dict) -> dict:
    for k, v in (other or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _deep_update(copy.deepcopy(cfg), overrides or {})
    # Variable de entorno para forzar modo simulado (sin modelos)
    if os.environ.get("STORYBOOK_MOCK") == "1":
        cfg["mock"] = True
    return cfg


def image_preset(cfg: dict) -> dict:
    img = cfg["image"]
    preset = copy.deepcopy(img["presets"][img["preset"]])
    preset["name"] = img["preset"]
    return preset


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
