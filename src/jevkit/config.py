"""Typed YAML configuration for datasets, mixes, training, and serving.

Unknown keys raise an error so a typo fails loudly instead of silently falling
back to a default. Plugin-specific settings go in the ``kwargs`` mappings.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level must be a mapping.")
    return data


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def from_dict(cls: type, data: dict[str, Any] | None, where: str = "") -> Any:
    """Build a (possibly nested) dataclass from a mapping, rejecting unknown keys."""

    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(
            f"{where or cls.__name__}: unknown keys {unknown}; allowed {sorted(names)}"
        )
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        nested = _dataclass_in(hints[name])
        if nested and isinstance(value, dict):
            value = from_dict(nested, value, f"{where}.{name}" if where else name)
        kwargs[name] = value
    return cls(**kwargs)


def _dataclass_in(hint: Any) -> type | None:
    if dataclasses.is_dataclass(hint):
        return hint
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        for arg in typing.get_args(hint):
            if dataclasses.is_dataclass(arg):
                return arg
    return None


def resolve_path(value: str | Path, base: Path) -> Path:
    """Resolve config-relative paths, falling back to the project root and cwd."""

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for root in (base, PROJECT_ROOT, Path.cwd()):
        if (root / candidate).exists():
            return root / candidate
    return PROJECT_ROOT / candidate


# --------------------------------------------------------------------------- data


@dataclass
class SourceSpec:
    """Where rows come from. ``type`` picks a registered source.

    Built in: ``hf`` (datasets.load_dataset) and ``python`` (your own function).
    """

    type: str = "hf"
    # hf: arguments for datasets.load_dataset
    path: str | None = None  # hub id such as "dair-ai/emotion", or "csv" / "json" / "parquet"
    name: str | None = None  # hub config name
    revision: str | None = None  # pin a commit SHA for reproducible builds
    data_files: Any = None
    trust_remote_code: bool = False
    # python: "package.module:function" returning {split: rows}
    function: str | None = None
    # anything a plugin source needs
    kwargs: dict[str, Any] = field(default_factory=dict)
    # Splits pooled for train/dev/calibration. Rows here never reach test.
    train_splits: list[str] = field(default_factory=lambda: ["train"])
    # Official held-out splits; test is drawn from these, else from the train pool.
    test_splits: list[str] = field(default_factory=list)
    # Cap rows read per split (deterministic shuffle first) for very large datasets.
    max_rows: int | None = None

    def __post_init__(self) -> None:
        self.train_splits = _as_list(self.train_splits)
        self.test_splits = _as_list(self.test_splits)


@dataclass
class NoneOption:
    """Add a "none of the listed" option, sometimes correct and sometimes a distractor."""

    correct_rate: float = 0.15
    distractor_rate: float = 0.15
    description: str = "None of the listed options matches."


@dataclass
class TaskSpec:
    """How one row becomes a state, question, and option set."""

    label: str  # label column
    # Column name, "{template} over {columns}", or a mapping of state keys to either.
    state: Any = "text"
    # Fixed text, a column name, a "{column}" template, or a list of phrasings.
    question: Any = "Which option best describes the state?"
    kind: str = "choice"  # choice | score (ordered levels, never shuffled) | noul (yes/no)
    # Raw label -> option key. None: ClassLabel names, bools as yes/no, else str(value).
    label_names: Any = None
    # Option key -> description shown to the model. Order = canonical order.
    options: dict[str, str] | None = None
    drop_labels: list = field(default_factory=list)
    # More classes than fit in A-H: options per example, an int or list to pick from.
    options_per_example: Any = None
    hard_negatives: int = 0
    none_option: NoneOption | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"choice", "score", "noul"}:
            raise ValueError(f"task.kind must be choice, score, or noul, not {self.kind!r}")
        self.drop_labels = _as_list(self.drop_labels)


@dataclass
class SampleSpec:
    """Examples per split. ``null`` takes everything that remains."""

    train: int | None = None
    dev: int | None = 200
    calibration: int | None = 0
    test: int | None = 200
    balance: bool = True  # round-robin across gold labels


@dataclass
class DatasetSpec:
    name: str  # short id used in record ids and per-source metrics
    source: SourceSpec
    task: TaskSpec
    sample: SampleSpec = field(default_factory=SampleSpec)
    description: str = ""


def load_dataset_spec(path: str | Path, overrides: dict[str, Any] | None = None) -> DatasetSpec:
    data = load_yaml(path)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict):
            data.setdefault(section, {}).update(values)
        else:
            data[section] = values
    return from_dict(DatasetSpec, data, str(path))


@dataclass
class TokenizerSpec:
    """Must match the model you train: its chat template shapes every prompt."""

    repo: str = "Qwen/Qwen3.5-4B"
    revision: str | None = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    chat_template_kwargs: dict[str, Any] = field(default_factory=lambda: {"enable_thinking": False})


@dataclass
class MixSpec:
    """A training set: several datasets rendered with one tokenizer and prompt format."""

    name: str
    datasets: list = field(default_factory=list)  # paths, or {path, <section overrides>}
    records: list = field(default_factory=list)  # prebuilt records JSONL dirs to merge
    tokenizer: TokenizerSpec = field(default_factory=TokenizerSpec)
    prompt_format: str = "json"
    plugins: list[str] = field(default_factory=list)
    seed: int = 20260920
    max_tokens: int = 2048
    description: str = ""


def load_mix(path: str | Path) -> MixSpec:
    return from_dict(MixSpec, load_yaml(path), str(path))


# ----------------------------------------------------------------------- training


@dataclass
class ModelSpec:
    name: str = "Qwen/Qwen3.5-4B"
    revision: str | None = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    max_seq_length: int = 2048
    load_in_4bit: bool = False  # QLoRA; needed for 4B models on 16 GB GPUs


@dataclass
class LoraSpec:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_rslora: bool = False


@dataclass
class OptimSpec:
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 128
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_rows: int = 512
    seed: int = 42


@dataclass
class SmokeSpec:
    train_rows: int = 128
    dev_rows: int = 64
    max_steps: int = 10


@dataclass
class ModalTrainSpec:
    app_name: str = "jevkit-train"
    gpu: str = "L40S"
    timeout_hours: int = 8
    data_volume: str = "jevkit-data"
    model_volume: str = "jevkit-models"
    cache_volume: str = "jevkit-cache"


@dataclass
class TrainConfig:
    artifact_name: str
    trainer: str = "unsloth"  # registered trainer backend: unsloth | hf | your plugin
    plugins: list[str] = field(default_factory=list)
    model: ModelSpec = field(default_factory=ModelSpec)
    lora: LoraSpec = field(default_factory=LoraSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    smoke: SmokeSpec = field(default_factory=SmokeSpec)
    modal: ModalTrainSpec = field(default_factory=ModalTrainSpec)
    kwargs: dict[str, Any] = field(default_factory=dict)  # extra settings for plugin trainers

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_train_config(path: str | Path) -> TrainConfig:
    return from_dict(TrainConfig, load_yaml(path), str(path))


# ------------------------------------------------------------------------ serving


@dataclass
class ServeConfig:
    artifact_name: str  # folder in the model volume written by training
    served_model_name: str
    app_name: str = "jevkit-serve"
    gpu: str = "L40S"
    model_volume: str = "jevkit-models"
    cache_volume: str = "jevkit-cache"
    api_secret: str = "jevkit-api"  # Modal secret holding VLLM_API_KEY
    max_model_len: int = 2048
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.90
    max_concurrent_inputs: int = 32
    min_containers: int = 0
    max_containers: int = 1
    scaledown_minutes: int = 5
    vllm_version: str = "0.30.0"
    reasoning_parser: str | None = "qwen3"  # null for non-Qwen3 models
    chat_template_kwargs: dict[str, Any] = field(default_factory=lambda: {"enable_thinking": False})
    # Serve the tokenizer written by `jevkit serve prepare` instead of the merged one.
    use_serve_tokenizer: bool = True
    # AutoTokenizer kwargs for `prepare`; Qwen3.5 needs fix_mistral_regex.
    tokenizer_kwargs: dict[str, Any] = field(default_factory=lambda: {"fix_mistral_regex": True})
    extra_vllm_args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_serve_config(path: str | Path) -> ServeConfig:
    return from_dict(ServeConfig, load_yaml(path), str(path))
