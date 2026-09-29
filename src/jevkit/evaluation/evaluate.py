"""Exact-choice accuracy and latency of an endpoint on a records file (the holdout)."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from jevkit.evaluation.metrics import percentile
from jevkit.inference.client import DecisionClient
from jevkit.io import read_jsonl


async def evaluate(
    client: DecisionClient, records: Path, output: Path, limit: int = 0, concurrency: int = 16
) -> dict[str, Any]:
    rows = read_jsonl(records)
    rows = rows if limit <= 0 else rows[:limit]
    if not rows:
        raise ValueError(f"{records} contains no rows.")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        labels = [o["label"] for o in row["options"]]
        async with semaphore:
            started = time.perf_counter()
            try:
                prediction, error = await client.adecide_record(row), None
            except Exception as exc:  # noqa: BLE001 - errors count as incorrect
                prediction, error = "", f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - started
        valid = prediction in labels
        return {
            "id": row["id"],
            "source": row["source"],
            "expected": row["answer"],
            "prediction": prediction,
            "valid": valid,
            "correct": valid and prediction == row["answer"],
            "latency_seconds": latency,
            "error": error,
        }

    cold = time.perf_counter()
    await client.async_.models.list()
    cold_start = time.perf_counter() - cold
    results = await asyncio.gather(*(one(row) for row in rows))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["source"]].append(result)
    latencies = [r["latency_seconds"] for r in results]
    correct = sum(r["correct"] for r in results)
    summary = {
        "model": client.model,
        "records": str(records),
        "count": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "invalid": sum(not r["valid"] for r in results),
        "errors": sum(r["error"] is not None for r in results),
        "cold_start_probe_seconds": cold_start,
        "latency_seconds": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "by_source": {
            source: {
                "count": len(items),
                "correct": sum(r["correct"] for r in items),
                "accuracy": sum(r["correct"] for r in items) / len(items),
            }
            for source, items in sorted(grouped.items())
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run(client: DecisionClient, records: Path, output: Path, limit: int, concurrency: int) -> dict:
    return asyncio.run(evaluate(client, records, output, limit, concurrency))
