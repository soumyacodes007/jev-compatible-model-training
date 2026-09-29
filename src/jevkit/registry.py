"""Plugin registries. Every swappable part of jevkit is looked up by name here.

Built-in components register themselves on import. Your own code registers the
same way, and is loaded from any of:

- a ``plugins: [my_package.my_module]`` list in a mix or train config
- the ``JEVKIT_PLUGINS`` environment variable (comma separated modules)
- a ``jevkit.plugins`` entry point in your package's metadata

Example::

    from jevkit.registry import sources

    @sources.register("tickets_db")
    def load_tickets(spec):
        return {"train": [...rows...], "test": [...rows...]}
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str, doc: str):
        self.kind = kind
        self.doc = doc
        self._items: dict[str, T] = {}

    def register(self, name: str, obj: T | None = None, *, replace: bool = False) -> Any:
        """Register ``obj`` under ``name``; usable directly or as a decorator."""

        def add(item: T) -> T:
            if name in self._items and not replace:
                raise ValueError(f"{self.kind} {name!r} is already registered.")
            self._items[name] = item
            return item

        return add(obj) if obj is not None else add

    def get(self, name: str) -> T:
        _load_builtins()
        if name not in self._items:
            raise KeyError(
                f"Unknown {self.kind} {name!r}. Registered: {sorted(self._items)}. "
                "Load plugins with `plugins:` in the config or JEVKIT_PLUGINS."
            )
        return self._items[name]

    def names(self) -> list[str]:
        _load_builtins()
        return sorted(self._items)


# Where rows come from: (DatasetSpec) -> {split: rows}. Rows are a
# datasets.Dataset or any iterable of dicts.
sources: Registry[Callable[..., dict[str, Any]]] = Registry(
    "source", "Loads raw rows per split for a dataset spec."
)
# How a decision is shown to the model: (state, question, options) -> messages.
prompt_formats: Registry[Callable[..., list[dict[str, str]]]] = Registry(
    "prompt format", "Renders one decision as chat messages."
)
# How a model is fine-tuned: (TrainConfig, data_dir, output_root, mode, overwrite) -> manifest.
trainers: Registry[Callable[..., dict[str, Any]]] = Registry(
    "trainer", "Fine-tunes a model on prompt/completion JSONL."
)

REGISTRIES = {"sources": sources, "prompt_formats": prompt_formats, "trainers": trainers}

_BUILTINS = (
    "jevkit.sources.hf",
    "jevkit.sources.python",
    "jevkit.prompts",
    "jevkit.training.backends",
)
_loaded: set[str] = set()


def _load_builtins() -> None:
    if "__builtins__" in _loaded:
        return
    _loaded.add("__builtins__")
    for module in _BUILTINS:
        importlib.import_module(module)
    load_plugins(os.environ.get("JEVKIT_PLUGINS", "").split(","))
    for entry in importlib.metadata.entry_points(group="jevkit.plugins"):
        if entry.value not in _loaded:
            _loaded.add(entry.value)
            entry.load()


def load_plugins(modules: Iterable[str] | None) -> None:
    """Import plugin modules so their ``register`` calls run."""

    import sys

    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))  # project-local plugin modules
    for module in modules or ():
        module = module.strip()
        if module and module not in _loaded:
            _loaded.add(module)
            importlib.import_module(module)
