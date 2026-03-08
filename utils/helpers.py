"""
General-purpose utility helpers.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def round_price(price: float, tick_size: float = 0.01) -> float:
    """Round a price to the nearest tick."""
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 10)


def round_qty(qty: float, step_size: float = 0.001) -> float:
    """Truncate quantity to exchange step size."""
    if step_size <= 0:
        return qty
    precision = max(0, int(round(-np.log10(step_size))))
    return float(f"{qty:.{precision}f}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Avoid ZeroDivisionError."""
    return numerator / denominator if denominator != 0 else default


def pct_change(new: float, old: float) -> float:
    return safe_divide(new - old, old)


def hash_dataframe(df: pd.DataFrame) -> str:
    """Deterministic hash of a DataFrame (for cache invalidation)."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def to_serializable(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to JSON-safe types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    return obj


def candle_to_dict(row: pd.Series) -> dict:
    return {
        "timestamp": row.name.isoformat() if hasattr(row.name, "isoformat") else str(row.name),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has required OHLCV columns and no NaN in them."""
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")
    df = df.dropna(subset=required)
    df = df[df["volume"] > 0]
    return df
