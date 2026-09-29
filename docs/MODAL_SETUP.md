# Modal setup

[Modal](https://modal.com) runs the GPU parts of jevkit: training and the vLLM
endpoint. Building datasets runs on your own machine and needs no GPU.

You do this once per machine and workspace.

## 1. Get a working shell

jevkit runs on **Linux or macOS**. On **Windows, use WSL** (Ubuntu). PowerShell
syntax is different (`&&`, `VAR=value cmd`, and `<...>` all fail there), and
Windows Smart App Control can block the compiled libraries `datasets` needs.

```bash
# Windows only: open PowerShell and enter Ubuntu
wsl
```

Everything below runs in that Linux shell. If you have not installed uv yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Install jevkit

```bash
cd /path/to/jev-compatible-model-training      # on WSL: /mnt/c/Users/<you>/...
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/jevkit   # keep the venv on the Linux disk (faster)
uv sync --extra data --extra dev
uv run jevkit --help
```

Put the `export` line in `~/.bashrc` so every new shell picks it up:

```bash
echo 'export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/jevkit' >> ~/.bashrc
```

## 3. Log in to Modal

Either open a browser login:

```bash
uv run modal setup
```

or use a token from **modal.com → Settings → API Tokens → New Token**:

```bash
uv run modal token set --token-id ak-... --token-secret as-...
```

Paste your own values, without `<` `>`. Treat the secret like a password: never
commit it or paste it into chats or issues. If it leaks, delete the token in
the dashboard and create a new one.

Check it:

```bash
uv run modal profile current      # prints your workspace name
```

## 4. Enable GPUs (payment method)

New workspaces get **$30 of free credit per month**, but Modal will not start
*any* GPU (even a T4) until a payment method is on file. Without one, runs stop
with:

```text
InvalidError: Please add a payment method to use L40S GPU functions.
```

Fix: **modal.com → Settings → Usage & Billing → add a payment method**. Then
set a **workspace spend limit** on the same page (e.g. $30) so usage can never
go past the free credit. Adding a card does not use the credit.

Rough costs for the default recipe (Qwen3.5-4B, L40S at about $2/hour):

| Step | GPU time | Cost |
| --- | --- | --- |
| Smoke run (10 steps; mostly model download) | 10–15 min | ~$0.50 |
| Full run, ~6k examples | 25–35 min | ~$1–1.50 |
| Full run, ~35k examples | ~2 h | ~$4 |
| Endpoint | only while in use (scales to zero after 5 idle min) | cents per session |

The first run also builds the training image (a few minutes, CPU only). Later
runs reuse it.

## 5. Create the endpoint API key (for serving)

The vLLM endpoint needs a key clients must send. Generate one locally and store
it as a Modal secret. `.env.modal` is git-ignored.

```bash
uv run python -c "import secrets; print('VLLM_API_KEY=' + secrets.token_urlsafe(32))" > .env.modal
chmod 600 .env.modal
uv run modal secret create jevkit-api --from-dotenv .env.modal
```

The name must match `api_secret` in `configs/serve/*.yaml`.

## 6. Storage

Volumes are created automatically on first use:

| Volume | Holds |
| --- | --- |
| `jevkit-data` | uploaded `train.jsonl` / `dev.jsonl` per build (never `test`) |
| `jevkit-models` | `<artifact>/adapter`, `<artifact>/merged`, `manifest.json` |
| `jevkit-cache` | Hugging Face and vLLM caches (faster restarts) |

Names are set in the `modal:` block of the train config and in the serve
configs. Change them to keep projects apart.

## 7. First run

```bash
uv run jevkit data build configs/mixes/emotion.yaml
uv run jevkit train modal --data data/processed/emotion-only --mode smoke
```

The command prints a link to the run in the Modal dashboard. When the smoke run
passes (it trains, evaluates, saves an adapter, and exports merged weights),
start the real run:

```bash
uv run jevkit train modal --data data/processed/emotion-only --mode full
```

Then serve it. See [TRAINING.md](TRAINING.md#serve-and-evaluate).

## Everyday commands

```bash
uv run modal app list                          # running / deployed apps
uv run modal app logs jevkit-serve             # endpoint logs
uv run modal app stop jevkit-serve             # take an endpoint down
uv run modal volume ls jevkit-models           # trained artifacts
uv run modal volume get jevkit-models jevkit-qwen35-4b/merged ./artifacts/merged   # download weights
uv run modal volume ls jevkit-data             # uploaded builds
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `The token '&&' is not a valid statement separator` | You are in PowerShell. Run `wsl` first. |
| `The '<' operator is reserved` | Replace `<id>` placeholders with real values, brackets included. |
| `Please add a payment method to use ... GPU functions` | Step 4. |
| `Token missing` / `not authenticated` | `uv run modal token set ...` inside the same shell (WSL and Windows keep separate tokens). |
| `Secret jevkit-api not found` | Step 5, or match `api_secret` in the serve config. |
| `.../merged does not exist` when serving | Run a **full** training first; smoke runs write `smoke-<artifact>-<time>/`. |
| `FileExistsError ... Pass --overwrite` | The artifact already exists; pass `--overwrite` or change `artifact_name`. |
| `DLL load failed ... Application Control policy` (Windows) | Use WSL (step 1). |
| First endpoint request takes minutes | Cold start: loading weights and vLLM warm-up. Later requests take about a second. |
