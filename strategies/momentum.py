"""
Momentum strategy.
Buys when price has been trending up with rising RSI and EMA alignment.
Sells when momentum reverses.
"""

from __future__ import annotations

import pandas as pd

from config.settings import settings
from strategies.base import BaseStrategy, Signal, SignalDirection
from utils.logger import get_logger

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Signal logic:
      BUY  – price > EMA20 > EMA50, RSI 50–70 (rising but not overbought),
              MACD histogram positive, volume surge
      SELL – price < EMA20 < EMA50, RSI 30–50 (falling but not oversold),
              MACD histogram negative
    """

    name = "momentum"

    def __init__(self) -> None:
        self.cfg = settings.strategy

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < 50:
            return Signal(symbol, SignalDirection.HOLD, 0.0, self.name)

        score = 0.0
        max_score = 5.0
        direction = SignalDirection.HOLD

        close = self._last(df, "close")
        ema20 = self._last(df, "ema_20")
        ema50 = self._last(df, "ema_50")
        rsi = self._last(df, "rsi")
        macd_hist = self._last(df, "macd_hist")
        momentum_10 = self._last(df, "momentum_10")
        volume_ratio = self._last(df, "volume_ratio", 1.0)

        # --- BUY conditions ---
        buy_score = 0.0
        if close > ema20:       buy_score += 1.0
        if ema20 > ema50:       buy_score += 1.0
        if 50 < rsi < 70:       buy_score += 1.0
        if macd_hist > 0:       buy_score += 1.0
        if momentum_10 > 0.02:  buy_score += 1.0
        if volume_ratio > 1.3:  buy_score += 0.5   # bonus

        # --- SELL conditions ---
        sell_score = 0.0
        if close < ema20:       sell_score += 1.0
        if ema20 < ema50:       sell_score += 1.0
        if 30 < rsi < 50:       sell_score += 1.0
        if macd_hist < 0:       sell_score += 1.0
        if momentum_10 < -0.02: sell_score += 1.0

        if buy_score >= 3.0 and buy_score > sell_score:
            direction = SignalDirection.BUY
            score = min(buy_score / max_score, 1.0)
        elif sell_score >= 3.0 and sell_score > buy_score:
            direction = SignalDirection.SELL
            score = min(sell_score / max_score, 1.0)

        logger.debug("[%s] Momentum signal: %s (score=%.2f)", symbol, direction.name, score)
        return Signal(
            symbol=symbol,
            direction=direction,
            strength=score,
            strategy_name=self.name,
            metadata={
                "rsi": rsi, "macd_hist": macd_hist,
                "momentum_10": momentum_10, "ema_trend": close > ema20,
            },
        )
