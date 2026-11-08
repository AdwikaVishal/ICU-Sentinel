"""Lightweight JSON experiment records for P2.

No external tracking service is used. Records are explicit about the data
source so synthetic software-validation runs cannot be mistaken for real
research evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert numpy scalars and NaN/inf values to strict-JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_experiment_record(
    record: dict,
    filepath: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write an experiment record without overwriting by default."""
    path = Path(filepath)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing experiment record: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(record), f, indent=2, allow_nan=False)
    return path
