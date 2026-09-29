from __future__ import annotations

import math


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
