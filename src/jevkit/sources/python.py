"""Your own loader: ``source: {type: python, function: "pkg.module:load"}``.

The function receives the ``source.kwargs`` mapping as keyword arguments and
returns ``{split_name: rows}`` (a DatasetDict works too). Use it for databases,
APIs, generated/synthetic data, or anything not on the Hub.
"""

from __future__ import annotations

import importlib
from typing import Any

from jevkit.registry import sources


def import_callable(target: str) -> Any:
    import sys
    from pathlib import Path

    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    module, _, attr = target.partition(":")
    if not attr:
        raise ValueError(f"Expected 'package.module:function', got {target!r}.")
    return getattr(importlib.import_module(module), attr)


@sources.register("python")
def load_python(spec: Any) -> Any:
    if not spec.source.function:
        raise ValueError(f"{spec.name}: source.function is required for type 'python'.")
    return import_callable(spec.source.function)(**spec.source.kwargs)
