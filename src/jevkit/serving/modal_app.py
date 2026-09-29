"""Serve a trained artifact with vLLM on Modal. Prefer the CLI wrapper::

    jevkit serve prepare --config configs/serve/l40s.yaml   # once per new artifact
    jevkit serve deploy  --config configs/serve/l40s.yaml

One serve config = one deployment (GPU, app name, artifact). The config is read
locally and baked into the image environment for the container.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

DEFAULT_CONFIG = "configs/serve/l40s.yaml"
VLLM_PORT = 8000


def _settings() -> dict:
    if modal.is_local():
        from jevkit.config import PROJECT_ROOT, load_serve_config, resolve_path

        path = resolve_path(os.environ.get("JEVKIT_SERVE_CONFIG", DEFAULT_CONFIG), PROJECT_ROOT)
        return load_serve_config(path).to_dict()
    return json.loads(os.environ["JEVKIT_SERVE_SETTINGS"])


S = _settings()
ARTIFACT_DIR = Path("/models") / S["artifact_name"]
MODEL_DIR = ARTIFACT_DIR / "merged"
TOKENIZER_DIR = ARTIFACT_DIR / "serve-tokenizer"

app = modal.App(S["app_name"])
model_volume = modal.Volume.from_name(S["model_volume"])
cache_volume = modal.Volume.from_name(S["cache_volume"], create_if_missing=True)
api_secret = modal.Secret.from_name(S["api_secret"], required_keys=["VLLM_API_KEY"])

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={S['vllm_version']}")
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "VLLM_CACHE_ROOT": f"/cache/vllm-{S['gpu'].lower().rstrip('!')}",
            # The slim image has no nvcc for FlashInfer's JIT sampler.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "JEVKIT_SERVE_SETTINGS": json.dumps(S),
        }
    )
)


@app.function(
    image=serve_image,
    volumes={"/models": model_volume},
    timeout=30 * 60,
)
def prepare_tokenizer() -> str:
    """Write a corrected serving tokenizer and patch the merged tokenizer config.

    Loads the merged tokenizer with ``tokenizer_kwargs`` (fix_mistral_regex for
    Qwen3.5), saves it beside the weights, and copies its config into the merged
    directory (backing up the original) because Qwen's multimodal processor also
    reads it from there. Harmless for models that do not need the fix.
    """

    import shutil

    from transformers import AutoTokenizer

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"{MODEL_DIR} does not exist. Run full training first.")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), **S["tokenizer_kwargs"])
    tokenizer.save_pretrained(str(TOKENIZER_DIR))
    backup = ARTIFACT_DIR / "original-tokenizer-config.json"
    if not backup.exists():
        shutil.copy(MODEL_DIR / "tokenizer_config.json", backup)
    shutil.copy(TOKENIZER_DIR / "tokenizer_config.json", MODEL_DIR / "tokenizer_config.json")
    model_volume.commit()
    return f"Wrote {TOKENIZER_DIR}"


@app.function(
    image=serve_image,
    gpu=S["gpu"],
    volumes={"/models": model_volume, "/cache": cache_volume},
    secrets=[api_secret],
    min_containers=S["min_containers"],
    max_containers=S["max_containers"],
    scaledown_window=S["scaledown_minutes"] * 60,
    timeout=30 * 60,
)
@modal.concurrent(max_inputs=S["max_concurrent_inputs"])
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * 60)
def serve() -> None:
    """Start the OpenAI-compatible vLLM server."""

    import subprocess

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"{MODEL_DIR} does not exist. Complete full training first.")
    tokenizer = TOKENIZER_DIR if S["use_serve_tokenizer"] and TOKENIZER_DIR.exists() else MODEL_DIR
    if S["use_serve_tokenizer"] and tokenizer == MODEL_DIR:
        print(
            f"WARNING: {TOKENIZER_DIR} missing; run `jevkit serve prepare`. Using merged tokenizer."
        )

    command = [
        "vllm",
        "serve",
        str(MODEL_DIR),
        "--tokenizer",
        str(tokenizer),
        "--served-model-name",
        S["served_model_name"],
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        os.environ["VLLM_API_KEY"],
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(S["max_model_len"]),
        "--gpu-memory-utilization",
        str(S["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(S["max_num_seqs"]),
        "--enable-prefix-caching",
    ]
    if S["reasoning_parser"]:
        command += ["--reasoning-parser", S["reasoning_parser"]]
    if S["chat_template_kwargs"]:
        command += ["--default-chat-template-kwargs", json.dumps(S["chat_template_kwargs"])]
    command += S["extra_vllm_args"]
    subprocess.Popen(command)
