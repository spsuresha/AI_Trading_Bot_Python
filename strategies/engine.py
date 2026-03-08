"""
Strategy engine – combines signals from all strategies using configurable weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from config.settings import settings
from features.engineer import FeatureEngineer
from strategies.base import BaseStrategy, Signal, SignalDirection
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.ai_prediction import AIPredictionStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


class CombinedSignal:
    """Aggregated signal result from the engine."""

    def __init__(
        self,
        symbol: str,
        combined_score: float,
        direction: SignalDirection,
        individual_signals: Dict[str, Signal],
    ) -> None:
        self.symbol = symbol
        self.combined_score = combined_score       # weighted sum, range [-1, 1]
        self.direction = direction
        self.individual_signals = individual_signals

    @property
    def is_actionable(self) -> bool:
        threshold = settings.strategy.signal_threshold
        return abs(self.combined_score) >= threshold

    def __repr__(self) -> str:
        return (
            f"CombinedSignal(symbol={self.symbol!r}, "
            f"direction={self.direction.name}, score={self.combined_score:.3f})"
        )


class StrategyEngine:
    """
    Runs all strategies and produces a combined weighted signal.

    Weights are configured in settings.strategy.weights.
    A final score in [-1, 0, +1] range is derived:
      score > threshold  → BUY
      score < -threshold → SELL
      otherwise          → HOLD
    """

    def __init__(self, extra_strategies: Optional[List[BaseStrategy]] = None) -> None:
        self.feature_eng = FeatureEngineer()
        self.strategies: Dict[str, BaseStrategy] = {
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
            "breakout": BreakoutStrategy(),
            "ai_prediction": AIPredictionStrategy(),
        }
        if extra_strategies:
            for strat in extra_strategies:
                self.strategies[strat.name] = strat

        self.weights: Dict[str, float] = settings.strategy.weights
        # Normalise weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info("StrategyEngine initialised with %d strategies", len(self.strategies))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process(self, df_raw: pd.DataFrame, symbol: str) -> CombinedSignal:
        """
        Run all strategies on the raw OHLCV DataFrame and return a CombinedSignal.

        df_raw is enriched with features internally; strategies receive the
        feature-enriched DataFrame.
        """
        df = self.feature_eng.compute_features(df_raw)

        individual: Dict[str, Signal] = {}
        for name, strategy in self.strategies.items():
            try:
                sig = strategy.generate_signal(df, symbol)
                individual[name] = sig
            except Exception as exc:
                logger.error("Strategy %s raised: %s", name, exc, exc_info=True)
                individual[name] = Signal(symbol, SignalDirection.HOLD, 0.0, name)

        combined_score = self._aggregate(individual)
        direction = self._score_to_direction(combined_score)

        result = CombinedSignal(
            symbol=symbol,
            combined_score=combined_score,
            direction=direction,
            individual_signals=individual,
        )
        logger.info(
            "[%s] Combined signal: %s (score=%.3f) | %s",
            symbol,
            direction.name,
            combined_score,
            {k: v.direction.name for k, v in individual.items()},
        )
        return result

    def process_all(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, CombinedSignal]:
        """Process all symbols in the provided data dict."""
        results: Dict[str, CombinedSignal] = {}
        for sym, df in data.items():
            results[sym] = self.process(df, sym)
        return results

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _aggregate(self, signals: Dict[str, Signal]) -> float:
        """
        Compute weighted sum of individual signal directions.
        Each directional value (-1/0/+1) is multiplied by the strategy's
        strength and weight, then summed.
        Returns a score in [-1, 1].
        """
        total = 0.0
        weight_used = 0.0
        for name, sig in signals.items():
            weight = self.weights.get(name, 0.0)
            value = float(sig.direction) * sig.strength
            total += weight * value
            weight_used += weight

        return total / weight_used if weight_used > 0 else 0.0

    def _score_to_direction(self, score: float) -> SignalDirection:
        threshold = settings.strategy.signal_threshold
        if score >= threshold:
            return SignalDirection.BUY
        elif score <= -threshold:
            return SignalDirection.SELL
        return SignalDirection.HOLD
