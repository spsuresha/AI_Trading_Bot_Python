"""
AI Prediction strategy.
Wraps the ML model predictor into the strategy interface.
"""

from __future__ import annotations

import pandas as pd

from models.predictor import SignalPredictor
from strategies.base import BaseStrategy, Signal, SignalDirection
from utils.logger import get_logger

logger = get_logger(__name__)


class AIPredictionStrategy(BaseStrategy):
    """
    Uses the trained XGBoost / RandomForest model to generate directional signals.
    The model's output probability is mapped to signal strength.
    """

    name = "ai_prediction"

    def __init__(self) -> None:
        self.predictor = SignalPredictor()
        if not self.predictor.is_ready:
            logger.warning("AI model not loaded. Run training first.")

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        if not self.predictor.is_ready:
            return Signal(symbol, SignalDirection.HOLD, 0.0, self.name)

        result = self.predictor.predict(df)
        raw_signal: int = result["signal"]          # -1 / 0 / +1
        probability: float = result["probability"]
        confidence: float = result["confidence"]

        if raw_signal == 1:
            direction = SignalDirection.BUY
        elif raw_signal == -1:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD

        logger.debug(
            "[%s] AI signal: %s (prob=%.4f, conf=%.4f)",
            symbol, direction.name, probability, confidence,
        )
        return Signal(
            symbol=symbol,
            direction=direction,
            strength=confidence if direction != SignalDirection.HOLD else 0.0,
            strategy_name=self.name,
            metadata={"probability": probability, "confidence": confidence},
        )
