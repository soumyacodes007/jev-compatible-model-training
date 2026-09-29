"""Hugging Face Hub (or local files through csv/json/parquet builders)."""

from __future__ import annotations

from typing import Any

from jevkit.registry import sources


@sources.register("hf")
def load_hf(spec: Any) -> Any:
    from datasets import load_dataset

    src = spec.source
    if not src.path:
        raise ValueError(f"{spec.name}: source.path is required for type 'hf'.")
    kwargs: dict[str, Any] = {"revision": src.revision}
    if src.name:
        kwargs["name"] = src.name
    if src.data_files is not None:
        kwargs["data_files"] = src.data_files
    if src.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return load_dataset(src.path, **kwargs, **src.kwargs)
