"""
Abstract base class for all trading strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional

import pandas as pd


class SignalDirection(IntEnum):
    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass
class Signal:
    symbol: str
    direction: SignalDirection          # -1 / 0 / +1
    strength: float = 0.0              # 0.0 → 1.0 (normalised conviction)
    strategy_name: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.direction != SignalDirection.HOLD and self.strength > 0.0


class BaseStrategy(ABC):
    """Every strategy must implement `generate_signal`."""

    name: str = "base"

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyse the OHLCV+features DataFrame and return a Signal.

        Parameters
        ----------
        df     : DataFrame with OHLCV and technical features
        symbol : trading pair string (e.g. "BTC/USDT")
        """

    def _last(self, df: pd.DataFrame, col: str, default=0.0):
        """Safely get the last value of a column."""
        try:
            return float(df[col].iloc[-1])
        except (KeyError, IndexError):
            return default
