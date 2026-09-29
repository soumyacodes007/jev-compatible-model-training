# Training, serving, and evaluation

## Pick a recipe

| Config | Trainer | GPU | When |
| --- | --- | --- | --- |
| `configs/train/qwen35_4b_lora.yaml` | `unsloth`, bf16 LoRA | L40S (48 GB) | Default; best quality/speed |
| `configs/train/qwen35_4b_qlora_t4.yaml` | `hf`, 4-bit QLoRA | T4 (16 GB) | Cheapest; also runs on Colab |

Copy one to start your own. The keys that matter most:

```yaml
artifact_name: my-model-v1     # output folder; bump per experiment
trainer: unsloth               # unsloth | hf | any registered plugin
model:
  name: Qwen/Qwen3.5-4B        # any Hugging Face chat model
  load_in_4bit: false
lora: {r: 8, alpha: 16}
optim: {epochs: 1.0, learning_rate: 5.0e-5, per_device_train_batch_size: 4,
        gradient_accumulation_steps: 2}
modal: {gpu: L40S}             # T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, ...
```

### Using a different base model

Change `model.name` in the train config **and** `tokenizer.repo` in the mix,
then rebuild the data. The prompt is rendered with that tokenizer's chat
template, so a build is tied to one model family. The build fails loudly if the
letters A–H are not single tokens for that tokenizer, or if the answer letter
merges into the prompt.

For non-Qwen3 models also set, in the mix `tokenizer.chat_template_kwargs: {}`,
and in the serve config `reasoning_parser: null`, `chat_template_kwargs: {}`,
and `tokenizer_kwargs: {}`.

## Run

```bash
# Modal (see MODAL_SETUP.md)
uv run jevkit train modal --config configs/train/qwen35_4b_lora.yaml \
    --data data/processed/<build> --mode smoke      # then --mode full

# Your own Linux + CUDA machine
uv sync --extra data --extra train                   # unsloth backend
uv sync --extra data --extra train-hf                # hf backend
uv run jevkit train local --config ... --data data/processed/<build> --mode smoke
```

- **smoke** trains 10 steps on 128 rows and writes `smoke-<artifact>-<timestamp>/`.
  Use it to prove the whole path works before paying for a full run.
- **full** trains on everything and writes `<artifact_name>/`:

```text
<artifact>/
  adapter/              LoRA weights + tokenizer
  merged/               full bf16 weights for vLLM
  manifest.json         config, data hashes, metrics, package versions, GPU
  data-manifest.json    the build manifest it was trained on
```

Only `instruction/train.jsonl` and `instruction/dev.jsonl` are uploaded. The
test split never leaves your machine, so evaluation stays honest.

## Serve and evaluate

```bash
uv run jevkit serve prepare --config configs/serve/l40s.yaml   # once per new artifact
uv run jevkit serve deploy  --config configs/serve/l40s.yaml
```

`serve prepare` writes a corrected serving tokenizer (needed for Qwen3.5,
harmless otherwise). `deploy` prints the URL, e.g.
`https://<workspace>--jevkit-serve-serve.modal.run`. Add `/v1` for clients.

The endpoint scales to zero after 5 idle minutes. The first request after that
waits for a cold start.

```bash
export JEVKIT_BASE_URL=https://<workspace>--jevkit-serve-serve.modal.run/v1
export JEVKIT_MODEL=jevkit-qwen35-4b               # served_model_name

uv run jevkit eval  --records data/processed/<build>/records/test.jsonl
uv run jevkit bench --records data/processed/<build>/records/test.jsonl \
    --label L40S --output data/results/bench.json
uv run jevkit chat
```

`eval` reports overall and per-source accuracy, invalid answers, errors, and
latency percentiles. It reads the prompt format from the build manifest
automatically.

### Calling the model from your code

```python
from jevkit.inference.client import DecisionClient

client = DecisionClient(base_url, api_key, model="jevkit-qwen35-4b", prompt_format="json")
client.decide(
    state={"customer_message": "The cash machine kept my card."},
    question="What is the customer asking about?",
    options=["Cash withdrawal fee", "Cash machine retained the card", "Card payment reversed"],
)
# {'answer': 'B', 'option': {...}, 'elapsed_seconds': 0.71}
```

Any OpenAI-compatible client works. Send the messages from
`jevkit.prompts.build_messages(...)` with `temperature=0` and
`extra_body={"structured_outputs": {"choice": ["A", "B", ...]}}` so the model
can only answer with a valid letter.

## Tuning tips

- **Look at `manifest.json` first**: letter balance, token lengths, and
  per-source counts explain most surprises.
- **Weak on one source?** Give it more `sample.train` in the mix, or write
  sharper option descriptions. They are the model's only definition of each
  class.
- **Ordered labels (ratings)** belong in `kind: score` so option order stays
  fixed.
- **Don't tune on the test split.** If test results shaped your changes,
  rebuild with a new `seed` or new test data before reporting numbers.
