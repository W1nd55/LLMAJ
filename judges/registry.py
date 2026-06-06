from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Type

import yaml

from .base import BaseJudge

JUDGE_REGISTRY: dict[str, Type[BaseJudge]] = {}


def register_judge(name: str):
    """Decorator to register a judge class under a given name."""

    def decorator(cls: Type[BaseJudge]):
        if name in JUDGE_REGISTRY:
            raise ValueError(
                f"Judge '{name}' is already registered by {JUDGE_REGISTRY[name].__name__}"
            )
        JUDGE_REGISTRY[name] = cls
        return cls

    return decorator


def list_judges() -> list[str]:
    """Return all registered judge names."""
    _ensure_models_imported()
    return list(JUDGE_REGISTRY.keys())


def get_judge(config_path: str) -> BaseJudge:
    """Instantiate a judge from a YAML config file.

    Config schema:
        judge:
          name: <registered name>
          model_path: <HF model id or local path>
          device: cuda
          dtype: bfloat16
          generation:
            temperature: 0
            max_new_tokens: 16384
            top_p: 0.9
    """
    _ensure_models_imported()

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    judge_cfg = cfg["judge"]
    name = judge_cfg["name"]

    if name not in JUDGE_REGISTRY:
        available = ", ".join(JUDGE_REGISTRY.keys()) or "(none)"
        raise ValueError(
            f"Unknown judge '{name}'. Available judges: {available}"
        )

    cls = JUDGE_REGISTRY[name]
    judge = cls(judge_cfg)
    judge.load_model()
    return judge


_models_imported = False


def _ensure_models_imported():
    """Auto-import all modules under judges.models to trigger registration."""
    global _models_imported
    if _models_imported:
        return
    models_dir = Path(__file__).parent / "models"
    for module_info in pkgutil.iter_modules([str(models_dir)]):
        importlib.import_module(f".models.{module_info.name}", package="judges")
    _models_imported = True
