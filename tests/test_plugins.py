"""Plugin system: custom sources, prompt formats, and trainers, all offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from test_pipeline import FakeTokenizer

from jevkit.config import TokenizerSpec, TrainConfig, load_dataset_spec
from jevkit.data.adapter import DatasetAdapter
from jevkit.data.build import build_mix
from jevkit.data.render import Renderer
from jevkit.inference.client import request_kwargs
from jevkit.io import read_jsonl
from jevkit.prompts import build_messages
from jevkit.registry import Registry, load_plugins, prompt_formats, sources, trainers
from jevkit.training import run_training

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = "examples.plugins.refund_policy"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)  # plugin modules resolve from the project root
    load_plugins([PLUGIN])


def test_builtins_and_plugins_are_registered():
    assert {"hf", "python", "refund_policy"} <= set(sources.names())
    assert {"json", "text", "bulleted"} <= set(prompt_formats.names())
    assert {"unsloth", "hf", "dry_run"} <= set(trainers.names())


def test_registry_rejects_duplicates_and_unknown_names():
    reg: Registry = Registry("thing", "test")
    reg.register("a", 1)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("a", 2)
    reg.register("a", 3, replace=True)
    with pytest.raises(KeyError, match="Unknown source"):
        sources.get("nope")


def test_plugin_source_feeds_the_generic_task_mapping():
    spec = load_dataset_spec(ROOT / "examples" / "refund_policy.yaml")
    adapter = DatasetAdapter(spec)
    train_pool, test_pool = adapter.examples()
    assert len(train_pool) == 600 and len(test_pool) == 120
    assert set(train_pool[0].state) == {"days_since_purchase", "item_opened", "amount_usd"}
    assert adapter.label_map.keys == ["approve", "partial", "deny", "escalate"]


def test_python_source_uses_a_plain_function():
    spec = load_dataset_spec(
        ROOT / "examples" / "refund_policy.yaml",
        {"source": {"type": "python", "function": f"{PLUGIN}:generate", "kwargs": {"n_train": 50}}},
    )
    train_pool, test_pool = DatasetAdapter(spec).examples()
    assert len(train_pool) == 50 and len(test_pool) == 120


def test_custom_prompt_format_reaches_training_and_inference(tmp_path):
    mix = tmp_path / "mix.yaml"
    mix.write_text(
        yaml.safe_dump(
            {
                "name": "plugin-demo",
                "plugins": [PLUGIN],
                "prompt_format": "bulleted",
                "datasets": [str(ROOT / "examples" / "refund_policy.yaml")],
            }
        )
    )
    manifest = build_mix(
        mix,
        output_root=tmp_path,
        renderer=Renderer(TokenizerSpec(), FakeTokenizer(), prompt_format="bulleted"),
    )
    assert manifest["prompt_format"] == "bulleted"
    prompt = read_jsonl(tmp_path / "plugin-demo" / "instruction" / "train.jsonl")[0]["prompt"]
    assert "Facts:\n- days_since_purchase:" in prompt

    record = read_jsonl(tmp_path / "plugin-demo" / "records" / "test.jsonl")[0]
    sent = request_kwargs("m", record, "bulleted")["messages"]
    assert sent == build_messages(
        record["state"], record["question"], record["options"], fmt="bulleted"
    )


def test_custom_trainer_is_dispatched_by_config(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(json.dumps({"prompt": "p", "completion": "A"}) + "\n")
    cfg = TrainConfig(artifact_name="demo", trainer="dry_run", plugins=[PLUGIN])
    manifest = run_training(cfg, data, tmp_path / "out", "smoke")
    assert manifest == {
        "artifact": "demo",
        "mode": "smoke",
        "trainer": "dry_run",
        "train_rows": 1,
        "invalid_rows": [],
    }
