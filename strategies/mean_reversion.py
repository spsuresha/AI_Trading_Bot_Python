"""
Mean-reversion strategy.
Buys when price is statistically cheap (low Z-score / lower Bollinger Band).
Sells when price is statistically expensive.
"""

from __future__ import annotations

import pandas as pd

from config.settings import settings
from strategies.base import BaseStrategy, Signal, SignalDirection
from utils.logger import get_logger

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Signal logic:
      BUY  – price below lower Bollinger Band, RSI < 35, Z-score < -2
      SELL – price above upper Bollinger Band, RSI > 65, Z-score > +2
    """

    name = "mean_reversion"

    def __init__(self) -> None:
        self.cfg = settings.strategy

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < 30:
            return Signal(symbol, SignalDirection.HOLD, 0.0, self.name)

        close = self._last(df, "close")
        rsi = self._last(df, "rsi")
        bb_pct = self._last(df, "bb_pct")        # 0 = lower band, 1 = upper band
        bb_upper = self._last(df, "bb_upper")
        bb_lower = self._last(df, "bb_lower")
        macd_hist = self._last(df, "macd_hist")

        # Z-score over last 20 candles
        window = df["close"].tail(20)
        z_score = (close - window.mean()) / (window.std() + 1e-9)

        score = 0.0
        direction = SignalDirection.HOLD

        # --- BUY (oversold / below mean) ---
        buy_score = 0.0
        if bb_pct < 0.1:                 buy_score += 2.0  # strong below lower band
        elif bb_pct < 0.2:               buy_score += 1.0
        if rsi < 35:                     buy_score += 1.5
        elif rsi < 40:                   buy_score += 0.5
        if z_score < -self.cfg.mean_rev_zscore_threshold:  buy_score += 2.0
        elif z_score < -1.5:             buy_score += 1.0
        if macd_hist > -0.001:           buy_score += 0.5  # MACD starting to recover

        # --- SELL (overbought / above mean) ---
        sell_score = 0.0
        if bb_pct > 0.9:                 sell_score += 2.0
        elif bb_pct > 0.8:               sell_score += 1.0
        if rsi > 65:                     sell_score += 1.5
        elif rsi > 60:                   sell_score += 0.5
        if z_score > self.cfg.mean_rev_zscore_threshold:   sell_score += 2.0
        elif z_score > 1.5:              sell_score += 1.0
        if macd_hist < 0.001:            sell_score += 0.5

        max_score = 6.0
        if buy_score >= 3.0 and buy_score > sell_score:
            direction = SignalDirection.BUY
            score = min(buy_score / max_score, 1.0)
        elif sell_score >= 3.0 and sell_score > buy_score:
            direction = SignalDirection.SELL
            score = min(sell_score / max_score, 1.0)

        logger.debug("[%s] MeanRev signal: %s (z=%.2f, bb_pct=%.2f)", symbol, direction.name, z_score, bb_pct)
        return Signal(
            symbol=symbol,
            direction=direction,
            strength=score,
            strategy_name=self.name,
            metadata={
                "z_score": round(z_score, 3),
                "bb_pct": round(bb_pct, 3),
                "rsi": round(rsi, 2),
            },
        )
