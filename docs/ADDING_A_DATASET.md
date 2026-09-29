# Adding a Hugging Face dataset

Every dataset is a YAML file in `configs/datasets/`. The generic adapter
(`src/jevkit/data/adapter.py`) turns each row into:

```json
{"state": ..., "question": "...", "options": [{"label": "A", "key": "...", "description": "..."}], "answer": "A"}
```

No Python changes are needed for a new classification dataset. For data that
is not on the Hub (databases, APIs, generated data) see [PLUGINS.md](PLUGINS.md).

## Workflow

```bash
uv run jevkit data inspect <hub-id> [--name <config>] --scaffold configs/datasets/<name>.yaml
# edit TODOs
uv run jevkit data preview configs/datasets/<name>.yaml   # prints options, split counts, sample records
uv run jevkit data build   configs/datasets/<name>.yaml   # or add it to a mix
```

`inspect` guesses the label column (a `ClassLabel`, or a column named
`label`/`answer`/`category`/...) and the state column (the longest text column),
and pins the dataset's current commit SHA.

## Reference

See `configs/datasets/_template.yaml` for every key with comments.

### `source` — where the data comes from

| Key | Meaning |
| --- | --- |
| `type` | `hf` (default), `python`, or a plugin source |
| `path` | Hub id (`dair-ai/emotion`) or a builder (`csv`, `json`, `parquet`) |
| `name` | Hub config name |
| `revision` | Commit SHA — pin it for reproducible builds |
| `data_files` | For local/remote files: `{train: a.csv, test: b.csv}` |
| `train_splits` | Pooled to draw train, dev, and calibration |
| `test_splits` | Official holdout. Empty = carve test out of the train pool |
| `max_rows` | Cap rows read per split (huge datasets) |
| `function` | For `type: python`: `package.module:function` returning `{split: rows}` |
| `kwargs` | Extra arguments for plugin/python sources |

### `task` — how a row becomes a decision

**state** — what the model reads:

```yaml
state: text                                   # one column
state: "{title}\n\n{body}"                    # template
state: {customer: message, account: tier}     # dict state (JSON in the prompt)
```

**question** — fixed text, a column name, a `{column}` template, or a list of
phrasings (one picked deterministically per row):

```yaml
question: "Using only the passage, classify this claim: {hypothesis}"
```

**label / label_names** — raw label → option key:

| Raw label | Default | Override |
| --- | --- | --- |
| `ClassLabel` int | its names | `label_names: [a, b, c]` |
| bool | `yes` / `no` | `label_names: {true: valid, false: invalid}` |
| string / other int | `str(value)` | mapping |

`drop_labels: [-1]` skips rows (e.g. unlabeled MNLI rows).

**options** — key → description the model sees. Their order is the canonical
order. Write real descriptions; they are the model's only definition of each
class. Omitted keys get a humanized default (`card_arrival` → `Card arrival.`).

**kind**

- `choice` — categorical; option order is shuffled per example.
- `score` — ordered levels (ratings); order is preserved.
- `noul` — exactly two labels (yes/no).

**More than 8 classes.** Letters are `A`–`H`, so sample distractors:

```yaml
options_per_example: [4, 6, 8]   # or a fixed int
hard_negatives: 2                # distractors sharing words with the gold key
none_option:                     # teach "none of these"
  correct_rate: 0.15             # gold removed, "none" is correct
  distractor_rate: 0.15          # "none" present but wrong
```

### `sample` — examples per split

```yaml
sample: {train: 2000, dev: 200, calibration: 0, test: 200, balance: true}
```

`null` takes everything left. `balance` round-robins over gold labels. Shortfalls
are reported in `manifest.json` under `audits.<name>.shortfalls`.

## Local files instead of the Hub

```yaml
name: tickets
source:
  type: hf
  path: csv
  data_files: {train: data/raw/tickets_train.csv, test: data/raw/tickets_test.csv}
task:
  state: {subject: subject, body: body}
  question: Which team should handle this ticket?
  label: team
```

## Output

`data/processed/<name>/`:

- `records/*.jsonl` — full records (used by `jevkit eval`)
- `instruction/*.jsonl` — `{"prompt", "completion"}` (used for training)
- `sft/*.jsonl` — `{"messages"}` for chat-format trainers
- `manifest.json` — counts, token stats, letter balance, hashes, audits
- `samples.json` — three training records per source

Only `instruction/train.jsonl` and `instruction/dev.jsonl` are uploaded for
training. `records/test.jsonl` is the untouched holdout.

## Beyond classification

The adapter covers anything that reduces to "pick one label for this row". For
generated or external data, either write a source plugin ([PLUGINS.md](PLUGINS.md))
or build decision records elsewhere and merge them with a mix's `records:`
entry. They are re-rendered with the mix's tokenizer and leakage-checked with
everything else.
