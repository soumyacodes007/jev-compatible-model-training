"""``jevkit``: build data, train, serve, and evaluate Jev-style decision models.

jevkit data inspect dair-ai/emotion --name split --scaffold configs/datasets/emotion.yaml
jevkit data preview configs/datasets/emotion.yaml
jevkit data build   configs/mixes/emotion.yaml
jevkit train modal  --data data/processed/emotion-only --mode smoke
jevkit serve prepare --config configs/serve/l40s.yaml
jevkit serve deploy  --config configs/serve/l40s.yaml
jevkit eval --records data/processed/emotion-only/records/test.jsonl
jevkit plugins
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from jevkit.config import PROJECT_ROOT
from jevkit.registry import load_plugins

TRAIN_CONFIG = "configs/train/qwen35_4b_lora.yaml"
SERVE_CONFIG = "configs/serve/l40s.yaml"


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------- data


def cmd_data_inspect(args: argparse.Namespace) -> None:
    from jevkit.data import inspect

    data = inspect.load(args.path, args.name, args.revision)
    _print(inspect.describe(data, args.rows))
    if args.scaffold:
        revision = args.revision or inspect.pinned_revision(args.path)
        out = Path(args.scaffold)
        if out.exists() and not args.overwrite:
            sys.exit(f"{out} exists; pass --overwrite.")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(inspect.scaffold(args.path, args.name, data, revision), encoding="utf-8")
        print(f"\nWrote {out}. Fill in the TODOs, then run: jevkit data preview {out}")


def cmd_data_preview(args: argparse.Namespace) -> None:
    """Build records for one dataset config in memory, without a tokenizer, and print some."""

    from jevkit.config import load_dataset_spec
    from jevkit.data.adapter import DatasetAdapter
    from jevkit.data.records import make_record, split_examples

    spec = load_dataset_spec(args.config)
    adapter = DatasetAdapter(spec)
    train_pool, test_pool = adapter.examples()
    splits, audit = split_examples(spec, train_pool, test_pool, seed=20260920)
    _print({"options": adapter.label_map.descriptions, "audit": audit})
    for example in splits["train"][: args.rows]:
        _print(make_record(spec, adapter.label_map, example, "train", 20260920))


def cmd_data_build(args: argparse.Namespace) -> None:
    from jevkit.data.build import build_mix

    kwargs = {"overwrite": args.overwrite}
    if args.output_root:
        kwargs["output_root"] = Path(args.output_root)
    manifest = build_mix(args.config, **kwargs)
    _print({split: info["count"] for split, info in manifest["splits"].items()})


# --------------------------------------------------------------------- train


def cmd_train_modal(args: argparse.Namespace) -> None:
    env = os.environ | {"JEVKIT_TRAIN_CONFIG": str(Path(args.config).resolve())}
    command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        "-m",
        "jevkit.training.modal_app",
        "--data",
        str(Path(args.data).resolve()),
        "--mode",
        args.mode,
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.skip_upload:
        command.append("--skip-upload")
    sys.exit(subprocess.call(command, env=env, cwd=PROJECT_ROOT))


def cmd_train_local(args: argparse.Namespace) -> None:
    from jevkit.config import load_train_config
    from jevkit.training import run_training

    data = Path(args.data)
    data = data / "instruction" if (data / "instruction").is_dir() else data
    _print(
        run_training(
            load_train_config(args.config), data, args.output_root, args.mode, args.overwrite
        )
    )


# --------------------------------------------------------------------- serve


def cmd_serve(args: argparse.Namespace) -> None:
    env = os.environ | {"JEVKIT_SERVE_CONFIG": str(Path(args.config).resolve())}
    target = "jevkit.serving.modal_app"
    if args.action == "deploy":
        command = [sys.executable, "-m", "modal", "deploy", "-m", target]
    else:
        command = [sys.executable, "-m", "modal", "run", "-m", f"{target}::prepare_tokenizer"]
    sys.exit(subprocess.call(command, env=env, cwd=PROJECT_ROOT))


# ------------------------------------------------------------ eval and bench


def _prompt_format(args: argparse.Namespace) -> str:
    """Explicit flag, else the format recorded in the build manifest, else json."""

    if args.prompt_format:
        return args.prompt_format
    records = getattr(args, "records", None)
    if records:
        manifest = Path(records).resolve().parent.parent / "manifest.json"
        if manifest.exists():
            return json.loads(manifest.read_text(encoding="utf-8")).get("prompt_format", "json")
    return "json"


def _client(args: argparse.Namespace):
    from dotenv import load_dotenv

    from jevkit.inference.client import DecisionClient

    load_dotenv(args.env_file)
    api_key = args.api_key or os.environ.get("VLLM_API_KEY")
    if not api_key:
        sys.exit(f"Set VLLM_API_KEY in {args.env_file} or pass --api-key.")
    base_url = args.base_url or os.environ.get("JEVKIT_BASE_URL")
    if not base_url:
        sys.exit("Pass --base-url (ending in /v1) or set JEVKIT_BASE_URL.")
    return DecisionClient(base_url, api_key, args.model, args.timeout, _prompt_format(args))


def cmd_eval(args: argparse.Namespace) -> None:
    from jevkit.evaluation import evaluate

    build = Path(args.records).resolve().parent.parent.name
    output = Path(args.output or f"data/results/{args.model}-{build}.json")
    _print(evaluate.run(_client(args), Path(args.records), output, args.limit, args.concurrency))


def cmd_bench(args: argparse.Namespace) -> None:
    from jevkit.evaluation import benchmark

    _print(
        benchmark.run(
            _client(args),
            Path(args.records),
            Path(args.output),
            args.label,
            args.requests,
            args.concurrency,
        )
    )


def cmd_chat(args: argparse.Namespace) -> None:
    client = _client(args)
    print(f"Model: {client.model}. Enter state (JSON or text), then .done on its own line.")

    def read_state() -> object:
        lines = []
        while (line := input("state> ")).strip() != ".done":
            lines.append(line)
        raw = "\n".join(lines).strip()
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return raw

    state = read_state()
    print("Commands: /state  /show  /quit")
    while True:
        question = input("question> ").strip()
        if question in {"/quit", "/exit"}:
            return
        if question == "/state":
            state = read_state()
            continue
        if question == "/show":
            _print(state)
            continue
        if not question:
            continue
        options: list[str] = []
        while len(options) < 8:
            text = input(f"option {len(options) + 1}> ").strip()
            if not text:
                if len(options) >= 2:
                    break
                print("Enter at least two options.")
                continue
            options.append(text)
        result = client.decide(state, question, options)
        chosen = result["option"]["description"] if result["option"] else "(invalid)"
        print(f"{result['answer']}: {chosen}  [{result['elapsed_seconds']:.3f}s]\n")


def cmd_plugins(args: argparse.Namespace) -> None:
    from jevkit.registry import REGISTRIES

    _print({name: registry.names() for name, registry in REGISTRIES.items()})


# -------------------------------------------------------------------- parser


def _endpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url", help="OpenAI-compatible URL ending in /v1 (or JEVKIT_BASE_URL)"
    )
    parser.add_argument("--model", default=os.environ.get("JEVKIT_MODEL", "jevkit-qwen35-4b"))
    parser.add_argument("--api-key")
    parser.add_argument("--env-file", default=".env.modal")
    parser.add_argument("--timeout", type=float, default=20 * 60)
    parser.add_argument("--prompt-format", help="default: read from the build manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevkit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plugins", default="", help="comma-separated plugin modules to import")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="inspect, preview, and build datasets").add_subparsers(
        dest="action", required=True
    )
    p = data.add_parser("inspect", help="show splits/features of a HF dataset")
    p.add_argument("path", help="hub id, e.g. dair-ai/emotion")
    p.add_argument("--name", help="hub config name")
    p.add_argument("--revision")
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--scaffold", help="write a starter dataset config to this path")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_data_inspect)

    p = data.add_parser("preview", help="dry-run one dataset config and print records")
    p.add_argument("config")
    p.add_argument("--rows", type=int, default=3)
    p.set_defaults(func=cmd_data_preview)

    p = data.add_parser("build", help="build a mix (or one dataset config) into data/processed")
    p.add_argument("config")
    p.add_argument("--output-root")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_data_build)

    train = sub.add_parser("train", help="fine-tune on Modal or a local GPU").add_subparsers(
        dest="action", required=True
    )
    for name, func in (("modal", cmd_train_modal), ("local", cmd_train_local)):
        p = train.add_parser(name)
        p.add_argument("--config", default=TRAIN_CONFIG)
        p.add_argument("--data", required=True, help="build dir, e.g. data/processed/emotion-only")
        p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
        p.add_argument("--overwrite", action="store_true")
        if name == "modal":
            p.add_argument("--skip-upload", action="store_true")
        else:
            p.add_argument("--output-root", default="artifacts")
        p.set_defaults(func=func)

    p = sub.add_parser("serve", help="prepare the tokenizer or deploy a vLLM endpoint on Modal")
    p.add_argument("action", choices=["prepare", "deploy"])
    p.add_argument("--config", default=SERVE_CONFIG)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="accuracy on a records file (use the test holdout)")
    _endpoint_args(p)
    p.add_argument("--records", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--output")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("bench", help="small warm latency benchmark")
    _endpoint_args(p)
    p.add_argument("--records", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("chat", help="interactive decisions against an endpoint")
    _endpoint_args(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("plugins", help="list registered sources, prompt formats, and trainers")
    p.set_defaults(func=cmd_plugins)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_plugins(args.plugins.split(","))
    args.func(args)


if __name__ == "__main__":
    main()
