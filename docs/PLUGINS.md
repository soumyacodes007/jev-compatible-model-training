# Plugins

Three parts of jevkit are swappable by name:

| Registry | Signature | Built in | Picked by |
| --- | --- | --- | --- |
| `sources` | `(spec) -> {split: rows}` | `hf`, `python` | `source.type` in a dataset config |
| `prompt_formats` | `(state, question, options) -> messages` | `json`, `text` | `prompt_format` in a mix |
| `trainers` | `(cfg, data_dir, output_root, mode, overwrite) -> manifest` | `unsloth`, `hf` | `trainer` in a train config |

See what is registered:

```bash
uv run jevkit plugins
uv run jevkit --plugins examples.plugins.refund_policy plugins
```

A complete working example lives in
[`examples/plugins/refund_policy.py`](../examples/plugins/refund_policy.py). It
adds one of each.

## Writing a plugin

A plugin is any importable Python module that calls `register`:

```python
# my_plugins/crm.py
from jevkit.registry import sources, prompt_formats, trainers

@sources.register("crm")
def crm_tickets(spec):
    rows = fetch_from_crm(**spec.source.kwargs)       # your code
    return {"train": rows[:-500], "test": rows[-500:]}  # lists of dicts or Datasets
```

Rows can have any columns. The dataset config's `task:` block maps them to
state, question, and label exactly as for Hugging Face data:

```yaml
name: crm
source:
  type: crm
  kwargs: {since: "2026-01-01"}
  train_splits: [train]
  test_splits: [test]
task:
  state: {subject: subject, body: body}
  question: Which team should handle this ticket?
  label: team
```

## Loading plugins

Any of these work, and they combine:

1. **In config:** `plugins: [my_plugins.crm]` in a mix or train config.
2. **Environment:** `export JEVKIT_PLUGINS=my_plugins.crm,my_plugins.other`
3. **CLI flag:** `uv run jevkit --plugins my_plugins.crm data build ...`
4. **Entry point** in your own installable package:

   ```toml
   [project.entry-points."jevkit.plugins"]
   crm = "my_plugins.crm"
   ```

Modules are imported from the current directory too, so a `my_plugins/`
folder next to your configs works without installing anything.

Trainer plugins listed in a train config's `plugins:` are also shipped into the
Modal container automatically (their top-level package is added to the image).

## Contracts

**Source.** Return `{split_name: rows}`. The split names must include the
config's `train_splits` and `test_splits`. Rows are a `datasets.Dataset` or an
iterable of dicts. Keep it deterministic (seed any randomness from `kwargs`) so
builds are reproducible.

No custom source needed? Use the built-in `python` source with any function:

```yaml
source: {type: python, function: "my_plugins.crm:load", kwargs: {since: "2026-01-01"}}
```

**Prompt format.** Return chat messages *without* the assistant answer; jevkit
appends it for training. The format name is stored in the build manifest.
Inference must use the same one: `jevkit eval` reads it automatically, and
`DecisionClient(prompt_format=...)` takes it explicitly. Keep option letters
visible to the model; the answer is always a letter.

**Trainer.** Read `data_dir/train.jsonl` (and `dev.jsonl` if present), each row
`{"prompt": ..., "completion": ...}`. Loss belongs on the completion only (the
answer letter plus EOS). Write weights under `output_root/<artifact>/` and
return a JSON-serializable manifest. For serving with jevkit's vLLM app, save
full merged weights to `<artifact>/merged/`. Reuse `jevkit.training.common`
(`prepare_run`, `sft_config`, `write_manifest`) to get smoke/full handling and
manifests for free.

## Beyond plugins

Everything else is plain config, so most changes need no code:

- new dataset → a YAML in `configs/datasets/` ([ADDING_A_DATASET.md](ADDING_A_DATASET.md))
- new model → `model.name` + mix `tokenizer.repo` ([TRAINING.md](TRAINING.md))
- new GPU, volume, or app name → `modal:` block / serve config
- prebuilt decision records → mix `records:` entry
