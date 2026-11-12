"""Small helpers shared across modules."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


class DataError(Exception):
    """Raised when an input file is missing, unreadable or unusable."""


def safe_div(a: float, b: float, default: float | None = None) -> float | None:
    """a / b, returning `default` when b is zero/NaN/None (no ZeroDivisionError)."""
    try:
        if b is None or a is None or pd.isna(a) or pd.isna(b) or b == 0:
            return default
        return float(a) / float(b)
    except (TypeError, ValueError):
        return default


def clean(obj: Any) -> Any:
    """Recursively convert numpy/pandas values into plain JSON-safe Python.

    NaN / inf become None (strict JSON has no NaN), floats are rounded.
    """
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if obj is pd.NA or obj is None:
        return None
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 4)
    return obj
