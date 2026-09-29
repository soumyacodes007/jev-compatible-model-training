"""Training entry point: dispatch to the configured trainer backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jevkit.config import TrainConfig, from_dict
from jevkit.registry import load_plugins, trainers


def run_training(
    config: TrainConfig | dict[str, Any],
    data_dir: str | Path,
    output_root: str | Path,
    mode: str = "smoke",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train with ``config.trainer`` (unsloth, hf, or a registered plugin)."""

    cfg = config if isinstance(config, TrainConfig) else from_dict(TrainConfig, config)
    load_plugins(cfg.plugins)
    return trainers.get(cfg.trainer)(cfg, data_dir, output_root, mode, overwrite)
