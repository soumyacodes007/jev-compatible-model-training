"""Offline tests: in-memory HF datasets and a character-level fake tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from jevkit.config import DatasetSpec, TokenizerSpec, from_dict, load_dataset_spec
from jevkit.data.adapter import DatasetAdapter
from jevkit.data.build import build_mix
from jevkit.data.records import make_record, split_examples
from jevkit.data.render import Renderer
from jevkit.io import read_jsonl
from jevkit.prompts import build_messages

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


def classlabel_data(n=60, names=("neg", "pos", "mid")) -> DatasetDict:
    features = Features({"text": Value("string"), "label": ClassLabel(names=list(names))})

    def rows(prefix, count):
        return {
            "text": [f"{prefix} text {i}" for i in range(count)],
            "label": [i % len(names) for i in range(count)],
        }

    return DatasetDict(
        {
            "train": Dataset.from_dict(rows("train", n), features=features),
            "test": Dataset.from_dict(rows("test", 20), features=features),
        }
    )


def spec(**task) -> DatasetSpec:
    return from_dict(
        DatasetSpec,
        {
            "name": "toy",
            "source": {"path": "toy", "train_splits": ["train"], "test_splits": ["test"]},
            "task": {"label": "label", "state": "text", "question": "Which?", **task},
            "sample": {"train": 30, "dev": 6, "calibration": 0, "test": 9},
        },
    )


def test_unknown_config_keys_fail():
    with pytest.raises(ValueError, match="unknown keys"):
        from_dict(
            DatasetSpec,
            {"name": "x", "source": {"path": "p", "splitz": []}, "task": {"label": "l"}},
        )


def test_classlabel_and_splits_do_not_leak():
    s = spec()
    adapter = DatasetAdapter(s, classlabel_data())
    train_pool, test_pool = adapter.examples()
    assert adapter.label_map.keys == ["neg", "pos", "mid"]
    splits, audit = split_examples(s, train_pool, test_pool, seed=1)
    assert audit["test_from"] == "official test splits"
    assert {len(v) for v in splits.values()} >= {30, 6, 9}
    groups = [{e.group for e in rows} for rows in splits.values()]
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            assert not a & b
    assert all(e.original_split == "test" for e in splits["test"])


def test_record_answer_matches_gold_and_score_keeps_order():
    s = spec(kind="score", options={"neg": "Negative.", "mid": "Mixed.", "pos": "Positive."})
    adapter = DatasetAdapter(s, classlabel_data())
    example = adapter.examples()[0][0]
    record = make_record(s, adapter.label_map, example, "train", seed=1)
    assert [o["key"] for o in record["options"]] == ["neg", "mid", "pos"]
    chosen = next(o for o in record["options"] if o["label"] == record["answer"])
    assert chosen["key"] == example.gold == record["answer_key"]


def test_many_classes_sample_distractors_and_none_option():
    names = [f"intent_{i}" for i in range(20)]
    s = spec(
        options_per_example=[4, 6],
        hard_negatives=2,
        none_option={"correct_rate": 0.5, "distractor_rate": 0.5},
    )
    adapter = DatasetAdapter(s, classlabel_data(80, names))
    for example in adapter.examples()[0][:40]:
        record = make_record(s, adapter.label_map, example, "train", seed=3)
        keys = [o["key"] for o in record["options"]]
        assert 4 <= len(keys) <= 7 and "none" in keys
        assert record["answer_key"] in (example.gold, "none")


def test_too_many_classes_without_sampling_errors():
    s = spec()
    adapter = DatasetAdapter(s, classlabel_data(40, [f"c{i}" for i in range(10)]))
    example = adapter.examples()[0][0]
    with pytest.raises(ValueError, match="options_per_example"):
        make_record(s, adapter.label_map, example, "train", seed=1)


def test_bool_labels_templates_and_mapping_state():
    data = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "passage": [f"passage {i}" for i in range(20)],
                    "q": [f"is {i} even" for i in range(20)],
                    "answer": [i % 2 == 0 for i in range(20)],
                }
            )
        }
    )
    s = from_dict(
        DatasetSpec,
        {
            "name": "bq",
            "source": {"path": "x", "test_splits": []},
            "task": {
                "kind": "noul",
                "label": "answer",
                "state": {"passage": "passage", "note": "row {q}"},
                "question": "Answer: {q}?",
            },
            "sample": {"train": None, "dev": 2, "test": 2},
        },
    )
    adapter = DatasetAdapter(s, data)
    train_pool, _ = adapter.examples()
    assert adapter.label_map.keys == ["yes", "no"]
    first = train_pool[0]
    assert first.state == {"passage": "passage 0", "note": "row is 0 even"}
    assert first.question == "Answer: is 0 even?" and first.gold == "yes"
    splits, audit = split_examples(s, train_pool, [], seed=1)
    assert audit["test_from"] == "train pool"
    assert len(splits["train"]) == 16  # None takes everything that remains


def test_string_labels_and_drop_labels():
    data = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": [f"t{i}" for i in range(12)],
                    "cat": ["b", "a", "skip"] * 4,
                }
            )
        }
    )
    s = from_dict(
        DatasetSpec,
        {"name": "s", "source": {"path": "x"}, "task": {"label": "cat", "drop_labels": ["skip"]}},
    )
    adapter = DatasetAdapter(s, data)
    pool, _ = adapter.examples()
    assert adapter.label_map.keys == ["a", "b"] and len(pool) == 8
    assert adapter.label_map.descriptions["a"] == "A."


def test_prompt_is_shared_between_training_and_inference():
    s = spec()
    adapter = DatasetAdapter(s, classlabel_data())
    example = adapter.examples()[0][0]
    record = make_record(s, adapter.label_map, example, "train", seed=1)
    renderer = Renderer(TokenizerSpec(), FakeTokenizer())
    instruction, chat, length = renderer.render(record)
    expected = build_messages(record["state"], record["question"], record["options"])
    assert chat["messages"][:-1] == expected
    assert instruction["completion"] == record["answer"] + "<eos>"
    assert length == len(instruction["prompt"]) + len(instruction["completion"])


def test_build_mix_end_to_end(tmp_path):
    ds_path = tmp_path / "toy.yaml"
    ds_path.write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "source": {"path": "toy"},
                "task": {"label": "label", "state": "text", "question": "Which?"},
                "sample": {"train": 20, "dev": 5, "test": 5},
            }
        )
    )
    mix_path = tmp_path / "mix.yaml"
    mix_path.write_text(
        yaml.safe_dump({"name": "m", "datasets": [str(ds_path)], "max_tokens": 4096})
    )
    s = load_dataset_spec(ds_path)
    manifest = build_mix(
        mix_path,
        output_root=tmp_path / "out",
        renderer=Renderer(TokenizerSpec(), FakeTokenizer()),
        adapters={"toy": DatasetAdapter(s, classlabel_data())},
    )
    out = tmp_path / "out" / "m"
    assert manifest["splits"]["train"]["count"] == 20
    assert len(read_jsonl(out / "instruction" / "train.jsonl")) == 20
    assert len(read_jsonl(out / "records" / "test.jsonl")) == 5
    assert json.loads((out / "manifest.json").read_text())["integrity"]["cross_split_overlap"] == 0
    with pytest.raises(FileExistsError):
        build_mix(mix_path, output_root=tmp_path / "out")


@pytest.mark.parametrize("path", sorted((ROOT / "configs" / "datasets").glob("*.yaml")))
def test_shipped_dataset_configs_parse(path):
    load_dataset_spec(path)
