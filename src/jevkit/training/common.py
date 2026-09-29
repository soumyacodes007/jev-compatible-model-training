"""Pieces shared by every trainer backend: data, run folders, SFT args, manifest."""

from __future__ import annotations

import importlib.metadata
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jevkit.config import TrainConfig
from jevkit.io import sha256_file, write_json


@dataclass
class RunPlan:
    cfg: TrainConfig
    mode: str
    data_dir: Path
    output: Path
    train_ds: Any
    dev_ds: Any
    max_steps: int
    warmup_steps: int
    logging_steps: int


def prepare_run(
    cfg: TrainConfig, data_dir: str | Path, output_root: str | Path, mode: str, overwrite: bool
) -> RunPlan:
    """Validate inputs, create the artifact folder, and load (smoke-sized) datasets."""

    from datasets import load_dataset

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    data_dir = Path(data_dir)
    train_path, dev_path = data_dir / "train.jsonl", data_dir / "dev.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} is missing. Build and upload the mix first.")

    stamp = f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    run_id = cfg.artifact_name if mode == "full" else f"smoke-{cfg.artifact_name}-{stamp}"
    output = Path(output_root) / run_id
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    o = cfg.optim
    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    dev_ds = (
        load_dataset("json", data_files=str(dev_path), split="train") if dev_path.exists() else None
    )
    if mode == "smoke":
        train_ds = train_ds.shuffle(seed=o.seed).select(
            range(min(cfg.smoke.train_rows, len(train_ds)))
        )
        dev_rows, steps, warmup, logging = cfg.smoke.dev_rows, cfg.smoke.max_steps, 1, 1
    else:
        dev_rows, steps, warmup, logging = o.eval_rows, o.max_steps, o.warmup_steps, o.logging_steps
    if dev_ds is not None:
        dev_ds = dev_ds.shuffle(seed=o.seed).select(range(min(dev_rows, len(dev_ds))))
    return RunPlan(cfg, mode, data_dir, output, train_ds, dev_ds, steps, warmup, logging)


def sft_config(plan: RunPlan, **overrides: Any) -> Any:
    """TRL SFTConfig from the train config. Loss is on the completion only."""

    from trl import SFTConfig

    o = plan.cfg.optim
    args = dict(
        output_dir=str(plan.output / "trainer"),
        max_length=plan.cfg.model.max_seq_length,
        completion_only_loss=True,
        packing=False,
        per_device_train_batch_size=o.per_device_train_batch_size,
        gradient_accumulation_steps=o.gradient_accumulation_steps,
        num_train_epochs=o.epochs,
        max_steps=plan.max_steps,
        learning_rate=o.learning_rate,
        warmup_steps=plan.warmup_steps,
        lr_scheduler_type=o.lr_scheduler_type,
        optim=o.optim,
        weight_decay=o.weight_decay,
        logging_steps=plan.logging_steps,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=o.seed,
    )
    args.update(overrides)
    return SFTConfig(**args)


def write_manifest(
    plan: RunPlan,
    train_result: Any,
    eval_metrics: dict,
    packages: list[str],
    extra: dict | None = None,
) -> dict[str, Any]:
    import torch

    o = plan.cfg.optim
    dev_path = plan.data_dir / "dev.jsonl"
    manifest = {
        "artifact": plan.output.name,
        "mode": plan.mode,
        "trainer": plan.cfg.trainer,
        "created_at": datetime.now(UTC).isoformat(),
        "config": plan.cfg.to_dict(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dataset": {
            "train_rows": len(plan.train_ds),
            "eval_rows": len(plan.dev_ds) if plan.dev_ds is not None else 0,
            "train_sha256": sha256_file(plan.data_dir / "train.jsonl"),
            "dev_sha256": sha256_file(dev_path) if dev_path.exists() else None,
        },
        "effective_batch_size": o.per_device_train_batch_size * o.gradient_accumulation_steps,
        "metrics": {"train": train_result.metrics, "eval": eval_metrics},
        "packages": {name: _version(name) for name in packages},
        "paths": {"adapter": str(plan.output / "adapter"), "merged": str(plan.output / "merged")},
        **(extra or {}),
    }
    mix_manifest = plan.data_dir / "manifest.json"
    if mix_manifest.exists():
        shutil.copy(mix_manifest, plan.output / "data-manifest.json")
    write_json(plan.output / "manifest.json", manifest)
    return manifest


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
