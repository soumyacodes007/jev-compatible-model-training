# jevkit

A pluggable framework for training **Jev-style decision models**: small LLMs
that read some state, answer one focused question, and pick exactly one option
from a fixed list.

```json
{"state": {"customer_message": "The cash machine kept my card."},
 "question": "What is the customer asking about?",
 "options": ["Cash withdrawal fee", "Cash machine retained the card", "Card payment reversed"]}
```

→ `B`

jevkit takes you from any dataset to a served, evaluated model:

```
dataset (HF Hub / files / your code) → build → LoRA fine-tune → vLLM endpoint → eval
```

- **Any dataset:** describe a Hugging Face dataset in YAML, with no code. For
  anything else, write a one-function source plugin.
- **Any model:** Qwen3.5-4B by default. Point at another Hugging Face chat model in config.
- **Any GPU:** Modal L40S by default, 4-bit QLoRA on a 16 GB T4, or your own CUDA box.
- **Pluggable:** sources, prompt formats, and trainers are registries you can extend.
- **Honest evaluation:** leakage-checked splits; the test set never leaves your machine.

## Quick start

Linux/macOS, or WSL on Windows. Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/soumyacodes007/jev-compatible-model-training.git
cd jev-compatible-model-training
uv sync --extra data --extra dev

# build a training set from a Hugging Face dataset
uv run jevkit data build configs/mixes/emotion.yaml

# train on Modal: smoke test first, then the real run
uv run jevkit train modal --data data/processed/emotion-only --mode smoke
uv run jevkit train modal --data data/processed/emotion-only --mode full

# serve and evaluate on the untouched holdout
uv run jevkit serve prepare && uv run jevkit serve deploy
uv run jevkit eval --base-url https://<workspace>--jevkit-serve-serve.modal.run/v1 \
    --records data/processed/emotion-only/records/test.jsonl
```

First time on Modal? Follow **[docs/MODAL_SETUP.md](docs/MODAL_SETUP.md)** (login,
payment method, API secret, costs, troubleshooting).

## Add your own dataset

```bash
uv run jevkit data inspect <org/dataset> --scaffold configs/datasets/mine.yaml
# fill in the question and option descriptions
uv run jevkit data preview configs/datasets/mine.yaml
uv run jevkit data build   configs/datasets/mine.yaml
```

```yaml
name: emotion
source: {type: hf, path: dair-ai/emotion, name: split, test_splits: [test]}
task:
  state: text
  question: Which emotion does the author express most strongly?
  label: label
  options:
    joy: Joy, happiness, or contentment.
    anger: Anger, irritation, or frustration.
    # ...
sample: {train: 6000, dev: 300, test: 600}
```

Multi-column states, `{column}` templates, label mapping, datasets with more
than 8 classes, "none of these" answers, and CSV/JSON files are covered in
**[docs/ADDING_A_DATASET.md](docs/ADDING_A_DATASET.md)**. Combine datasets into one
training set with a mix (`configs/mixes/`).

## Plug in your own parts

```python
from jevkit.registry import sources, prompt_formats, trainers

@sources.register("crm")                # source.type: crm
def crm(spec): ...                       # -> {split: rows}

@prompt_formats.register("compact")     # prompt_format: compact
def compact(state, question, options): ...   # -> chat messages

@trainers.register("my_trainer")        # trainer: my_trainer
def my_trainer(cfg, data_dir, output_root, mode, overwrite): ...  # -> manifest
```

Load plugins with `plugins: [...]` in a config, `JEVKIT_PLUGINS`, `--plugins`, or
a `jevkit.plugins` entry point. **[docs/PLUGINS.md](docs/PLUGINS.md)** covers the
contracts, and [`examples/`](examples/) has a working plugin with all three.

## Layout

```
configs/
  datasets/    one YAML per dataset (+ _template.yaml)
  mixes/       datasets combined into one training set
  train/       model, LoRA, optimizer, trainer, GPU presets
  serve/       one file per vLLM deployment
src/jevkit/
  registry.py  plugin registries
  prompts.py   built-in prompt formats
  sources/     hf, python
  data/        adapter, option sampling, splits, rendering, build, inspect
  training/    run_training dispatch, backends (unsloth, hf), Modal app
  serving/     vLLM on Modal
  inference/   OpenAI-compatible client
  evaluation/  holdout accuracy, latency benchmark
  cli.py       the `jevkit` command
examples/      plugin example + configs using it
docs/          Modal setup, training, datasets, plugins
tests/         offline tests (no network, no GPU)
```

## Commands

| Command | Does |
| --- | --- |
| `jevkit data inspect <hub-id> [--scaffold f.yaml]` | Show splits/columns; write a starter config |
| `jevkit data preview <dataset.yaml>` | Dry run: options, split counts, sample records |
| `jevkit data build <mix-or-dataset.yaml>` | Write `data/processed/<name>/` |
| `jevkit train modal\|local --data <build> --mode smoke\|full` | Fine-tune |
| `jevkit serve prepare\|deploy --config <serve.yaml>` | vLLM endpoint on Modal |
| `jevkit eval --records <build>/records/test.jsonl` | Accuracy + latency |
| `jevkit bench`, `jevkit chat` | Latency benchmark, interactive client |
| `jevkit plugins` | List registered sources, prompt formats, trainers |

## What a build guarantees

- Each normalized state is in exactly one split. Official test splits are claimed
  first, so they never leak into training.
- No id, state group, or exact prompt crosses splits (checked at build time).
- Options are shuffled per example (except `kind: score`), and answer letters
  stay balanced.
- The answer letter is one token and never merges with the prompt.
- Over-length rows are dropped and listed in `manifest.json`, never truncated.
- Builds refuse to overwrite unless you pass `--overwrite`.

## Tests

```bash
uv run python -m pytest -q
```

## Licenses

Every dataset and base model has its own license. Check them before you
redistribute built data or trained weights.
