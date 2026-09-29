"""Example plugin: one file that adds a source, a prompt format, and a trainer.

Load it from a config (``plugins: [examples.plugins.refund_policy]``), with
``JEVKIT_PLUGINS=examples.plugins.refund_policy``, or ``jevkit --plugins ...``.
Check it registered: ``jevkit --plugins examples.plugins.refund_policy plugins``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from jevkit.registry import prompt_formats, sources, trainers

# --------------------------------------------------------------------- source
# A source returns {split: rows}. Rows are plain dicts; the dataset config's
# `task:` block maps their columns to state / question / label as usual.


def generate(n_train: int = 600, n_test: int = 120, seed: int = 0) -> dict[str, list[dict]]:
    """Synthetic refund decisions with executable ground truth."""

    rng = random.Random(seed)

    def row() -> dict[str, Any]:
        days = rng.randint(0, 60)
        opened = rng.random() < 0.5
        amount = rng.randint(5, 900)
        if days > 30:
            label = "deny"
        elif amount > 500:
            label = "escalate"
        else:
            label = "approve" if not opened else "partial"
        return {
            "days_since_purchase": days,
            "item_opened": opened,
            "amount_usd": amount,
            "label": label,
        }

    return {"train": [row() for _ in range(n_train)], "test": [row() for _ in range(n_test)]}


@sources.register("refund_policy")
def refund_policy_source(spec: Any) -> dict[str, list[dict]]:
    return generate(**spec.source.kwargs)


# -------------------------------------------------------------- prompt format
# A prompt format returns chat messages. Training and inference must use the
# same one; the build manifest records it and `jevkit eval` reads it back.


@prompt_formats.register("bulleted")
def bulleted(state: Any, question: str, options: list[dict[str, str]]) -> list[dict[str, str]]:
    facts = state if isinstance(state, dict) else {"text": state}
    lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
    choices = "\n".join(f"{o['label']}) {o['description']}" for o in options)
    return [
        {"role": "system", "content": "Answer with a single option letter."},
        {"role": "user", "content": f"Facts:\n{lines}\n\n{question}\n\n{choices}"},
    ]


# -------------------------------------------------------------------- trainer
# A trainer gets (TrainConfig, data_dir, output_root, mode, overwrite) and
# returns a manifest dict. This one needs no GPU: it validates the data and
# writes a manifest, which is handy for testing a pipeline end to end.


@trainers.register("dry_run")
def dry_run(
    cfg: Any, data_dir: str | Path, output_root: str | Path, mode: str, overwrite: bool
) -> dict:
    data_dir = Path(data_dir)
    rows = [json.loads(line) for line in (data_dir / "train.jsonl").open(encoding="utf-8")]
    bad = [i for i, r in enumerate(rows) if not {"prompt", "completion"} <= r.keys()]
    manifest = {
        "artifact": cfg.artifact_name,
        "mode": mode,
        "trainer": "dry_run",
        "train_rows": len(rows),
        "invalid_rows": bad[:20],
    }
    out = Path(output_root) / f"dry-run-{cfg.artifact_name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
