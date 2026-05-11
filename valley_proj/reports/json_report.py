from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return out
