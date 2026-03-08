"""
Reinforcement Learning strategy — wraps the trained PPO agent.

The PPOInference engine is cached per symbol so the internal position
state stays consistent across consecutive calls from the strategy engine.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalDirection
from utils.logger import get_logger

logger = get_logger(__name__)


class RLStrategy(BaseStrategy):
    """
    Generates trading signals using a trained PPO reinforcement-learning agent.

    The agent observes a 24-dimensional normalised feature vector derived from
    the raw OHLCV data and outputs a discrete action: HOLD / BUY / SELL.
    Action confidence (argmax probability) is used as the signal strength.

    The `PPOInferenceCache` ensures the same `PPOInference` instance (and its
    internal portfolio-state) is reused across calls for the same symbol.
    """

    name = "rl_ppo"

    def __init__(self) -> None:
        # Import lazily so the strategy engine still loads even without PyTorch
        try:
            from rl_agent.inference import PPOInferenceCache
            self._cache = PPOInferenceCache
            self._available = True
        except Exception as exc:
            logger.warning("RLStrategy unavailable: %s", exc)
            self._cache = None
            self._available = False

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        if not self._available or self._cache is None:
            return Signal(symbol, SignalDirection.HOLD, 0.0, self.name)

        try:
            agent = self._cache.get(symbol)
            if not agent.is_ready:
                return Signal(symbol, SignalDirection.HOLD, 0.0, self.name)

            result = agent.predict(df)
            raw_signal: int = result["signal"]        # −1 / 0 / +1
            confidence: float = result["confidence"]
            action_name: str = result["action_name"]

            if raw_signal == 1:
                direction = SignalDirection.BUY
            elif raw_signal == -1:
                direction = SignalDirection.SELL
            else:
                direction = SignalDirection.HOLD

            logger.debug(
                "[%s] RL signal: %s (action=%s, confidence=%.4f)",
                symbol, direction.name, action_name, confidence,
            )
            return Signal(
                symbol=symbol,
                direction=direction,
                strength=confidence if direction != SignalDirection.HOLD else 0.0,
                strategy_name=self.name,
                metadata={
                    "action_name":     action_name,
                    "confidence":      confidence,
                    "portfolio_value": result.get("portfolio_value", 0.0),
                    "position":        result.get("position", 0),
                },
            )

        except Exception as exc:
            logger.warning("[%s] RLStrategy.generate_signal error: %s", symbol, exc)
            return Signal(symbol, SignalDirection.HOLD, 0.0, self.name)
