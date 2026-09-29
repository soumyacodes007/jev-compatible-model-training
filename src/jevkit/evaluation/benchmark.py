"""Small warm-latency and throughput benchmark against one endpoint."""

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


def balanced_rows(path: Path, count: int) -> list[dict[str, Any]]:
    """Deterministic rows in round-robin source order."""

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        by_source[row["source"]].append(row)
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < count and any(index < len(v) for v in by_source.values()):
        for source in sorted(by_source):
            if index < len(by_source[source]) and len(selected) < count:
                selected.append(by_source[source][index])
        index += 1
    if len(selected) != count:
        raise ValueError(f"Requested {count} rows, but {path} has only {len(selected)}.")
    return selected


async def benchmark(
    client: DecisionClient,
    records: Path,
    output: Path,
    label: str,
    requests: int,
    concurrency: int,
) -> dict[str, Any]:
    rows = balanced_rows(records, requests)

    async def checked(row: dict[str, Any]) -> float:
        started = time.perf_counter()
        answer = await client.adecide_record(row)
        if answer not in {o["label"] for o in row["options"]}:
            raise RuntimeError(f"The endpoint returned an invalid choice: {answer!r}")
        return time.perf_counter() - started

    await client.async_.models.list()
    await checked(rows[0])  # warm-up
    semaphore = asyncio.Semaphore(concurrency)

    async def measured(row: dict[str, Any]) -> float:
        async with semaphore:
            return await checked(row)

    started = time.perf_counter()
    latencies = await asyncio.gather(*(measured(row) for row in rows))
    elapsed = time.perf_counter() - started
    report = {
        "endpoint_label": label,
        "model": client.model,
        "measured_requests": len(rows),
        "warmup_requests": 1,
        "concurrency": concurrency,
        "metrics": {
            "p50_seconds": percentile(latencies, 0.5),
            "p95_seconds": percentile(latencies, 0.95),
            "throughput_requests_per_second": len(rows) / elapsed,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run(
    client: DecisionClient, records: Path, output: Path, label: str, requests: int, concurrency: int
) -> dict:
    return asyncio.run(benchmark(client, records, output, label, requests, concurrency))
