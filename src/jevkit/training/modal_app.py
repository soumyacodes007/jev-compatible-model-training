"""Train on a Modal GPU. Prefer the CLI wrapper::

    jevkit train modal --config configs/train/qwen35_4b_lora.yaml \
        --data data/processed/emotion-only --mode smoke

Direct use: ``JEVKIT_TRAIN_CONFIG=... modal run -m jevkit.training.modal_app --data ...``.

The local side reads the train config, uploads the build's train/dev files
(never the test holdout) to the data volume, and calls the remote function.
Plugin modules named in the config are shipped into the container too.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

DATA_MOUNT, MODEL_MOUNT, CACHE_MOUNT = "/data", "/models", "/cache"
DEFAULT_CONFIG = "configs/train/qwen35_4b_lora.yaml"
_KEYS = ("app_name", "gpu", "timeout_hours", "data_volume", "model_volume", "cache_volume")


def _settings() -> dict[str, str]:
    """Decorator settings: from the YAML locally, from baked image env in the container."""

    if modal.is_local():
        from jevkit.config import PROJECT_ROOT, load_train_config, resolve_path

        path = resolve_path(os.environ.get("JEVKIT_TRAIN_CONFIG", DEFAULT_CONFIG), PROJECT_ROOT)
        cfg = load_train_config(path)
        settings = {key: str(getattr(cfg.modal, key)) for key in _KEYS}
        settings["plugins"] = ",".join(cfg.plugins)
        return settings
    return {key: os.environ[f"JEVKIT_MODAL_{key.upper()}"] for key in (*_KEYS, "plugins")}


SETTINGS = _settings()
app = modal.App(SETTINGS["app_name"])
data_volume = modal.Volume.from_name(SETTINGS["data_volume"], create_if_missing=True)
model_volume = modal.Volume.from_name(SETTINGS["model_volume"], create_if_missing=True)
cache_volume = modal.Volume.from_name(SETTINGS["cache_volume"], create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(
        "unsloth[cu128-torch2100]==2026.9.11",
        "unsloth-zoo==2026.9.7",
        "transformers==5.5.0",
        "trl==0.24.0",
        "datasets==4.3.0",
        "peft",
        "bitsandbytes",
        "pyyaml",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_MOUNT}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "true",
            **{f"JEVKIT_MODAL_{k.upper()}": v for k, v in SETTINGS.items()},
        }
    )
    .add_local_python_source(
        "jevkit", *{p.split(".")[0] for p in SETTINGS["plugins"].split(",") if p}
    )
)


@app.function(
    image=train_image,
    gpu=SETTINGS["gpu"],
    volumes={DATA_MOUNT: data_volume, MODEL_MOUNT: model_volume, CACHE_MOUNT: cache_volume},
    timeout=int(SETTINGS["timeout_hours"]) * 60 * 60,
    max_containers=1,
    retries=0,
    single_use_containers=True,
)
def train(config: dict, dataset: str, mode: str = "smoke", overwrite: bool = False) -> dict:
    from jevkit.training import run_training

    data_volume.reload()
    manifest = run_training(config, Path(DATA_MOUNT) / dataset, MODEL_MOUNT, mode, overwrite)
    model_volume.commit()
    cache_volume.commit()
    return manifest


def upload_build(data_dir: Path) -> str:
    """Upload train/dev/manifest of a build; returns its folder name on the volume."""

    data_dir = Path(data_dir)
    files = {
        "train.jsonl": data_dir / "instruction" / "train.jsonl",
        "dev.jsonl": data_dir / "instruction" / "dev.jsonl",
        "manifest.json": data_dir / "manifest.json",
    }
    if not files["train.jsonl"].exists():
        raise FileNotFoundError(f"{files['train.jsonl']} not found. Run `jevkit data build` first.")
    with data_volume.batch_upload(force=True) as batch:
        for remote, local in files.items():
            if local.exists():
                batch.put_file(str(local), f"/{data_dir.name}/{remote}")
    print(f"Uploaded {data_dir.name} (train/dev only) to volume {SETTINGS['data_volume']}")
    return data_dir.name


@app.local_entrypoint()
def main(data: str, mode: str = "smoke", overwrite: bool = False, skip_upload: bool = False):
    import json

    from jevkit.config import PROJECT_ROOT, load_train_config, resolve_path

    cfg = load_train_config(
        resolve_path(os.environ.get("JEVKIT_TRAIN_CONFIG", DEFAULT_CONFIG), PROJECT_ROOT)
    )
    name = Path(data).name if skip_upload else upload_build(Path(data))
    print(json.dumps(train.remote(cfg.to_dict(), name, mode, overwrite), indent=2, default=str))
