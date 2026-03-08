"""
Breakout strategy.
Detects price breaks above recent highs (bullish) or below recent lows (bearish)
accompanied by a volume surge.
"""

from __future__ import annotations

import pandas as pd

from config.settings import settings
from strategies.base import BaseStrategy, Signal, SignalDirection
from utils.logger import get_logger

logger = get_logger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    Signal logic:
      BUY  – close > highest high of last N candles (excluding current),
              volume > 1.5x average, ATR expanding
      SELL – close < lowest low of last N candles (excluding current),
              volume > 1.5x average, ATR expanding
    """

    name = "breakout"

    def __init__(self) -> None:
        self.cfg = settings.strategy

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        lookback = self.cfg.breakout_lookback
        if len(df) < lookback + 5:
            return Signal(symbol, SignalDirection.HOLD, 0.0, self.name)

        recent = df.iloc[-(lookback + 1):-1]   # last N candles, exclude current
        current = df.iloc[-1]

        close = float(current["close"])
        high_n = float(recent["high"].max())
        low_n = float(recent["low"].min())
        volume_ratio = self._last(df, "volume_ratio", 1.0)
        atr = self._last(df, "atr")
        atr_prev = float(df["atr"].iloc[-5:-1].mean()) if "atr" in df.columns else atr
        atr_expanding = atr > atr_prev * 1.05

        score = 0.0
        direction = SignalDirection.HOLD

        # --- BULLISH breakout ---
        if close > high_n:
            score += 0.5
            if volume_ratio > 1.5:    score += 0.3
            if volume_ratio > 2.0:    score += 0.1  # extra for strong volume
            if atr_expanding:         score += 0.1
            direction = SignalDirection.BUY

        # --- BEARISH breakout ---
        elif close < low_n:
            score += 0.5
            if volume_ratio > 1.5:    score += 0.3
            if volume_ratio > 2.0:    score += 0.1
            if atr_expanding:         score += 0.1
            direction = SignalDirection.SELL

        score = min(score, 1.0)
        if score < 0.5:
            direction = SignalDirection.HOLD
            score = 0.0

        logger.debug(
            "[%s] Breakout signal: %s (close=%.2f, H%d=%.2f, L%d=%.2f, vol_ratio=%.2f)",
            symbol, direction.name, close, lookback, high_n, lookback, low_n, volume_ratio,
        )
        return Signal(
            symbol=symbol,
            direction=direction,
            strength=score,
            strategy_name=self.name,
            metadata={
                "high_n": round(high_n, 4),
                "low_n": round(low_n, 4),
                "volume_ratio": round(volume_ratio, 2),
                "atr_expanding": atr_expanding,
            },
        )
