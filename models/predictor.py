"""
Signal predictor – wraps the trained model for inference.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings
from features.engineer import FeatureEngineer
from models.trainer import ModelTrainer
from utils.logger import get_logger

logger = get_logger(__name__)


class SignalPredictor:
    """
    Loads the saved model and produces trading signals for a live DataFrame.

    Signal values:
        +1  → BUY  (model predicts price will rise)
        -1  → SELL / SHORT
         0  → HOLD (probability too close to 0.5)
    """

    CONFIDENCE_THRESHOLD = 0.55   # min prob to emit a signal

    def __init__(self) -> None:
        self.trainer = ModelTrainer()
        self.feature_eng = FeatureEngineer()
        self._loaded = self.trainer.load()

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.trainer.model is not None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Generate a signal from the most recent candle.

        Parameters
        ----------
        df : raw OHLCV DataFrame (will be feature-engineered internally)

        Returns
        -------
        dict with keys: signal (-1/0/1), probability, confidence
        """
        if not self.is_ready:
            logger.warning("Model not loaded – returning neutral signal")
            return {"signal": 0, "probability": 0.5, "confidence": 0.0}

        df_feat = self.feature_eng.compute_features(df)
        if df_feat.empty:
            return {"signal": 0, "probability": 0.5, "confidence": 0.0}

        # Use the last row for inference
        feature_cols = self.trainer.feature_columns
        available = [c for c in feature_cols if c in df_feat.columns]
        missing = [c for c in feature_cols if c not in df_feat.columns]

        if missing:
            logger.warning("Missing feature columns: %s", missing)
            for col in missing:
                df_feat[col] = 0.0

        X = df_feat[feature_cols].iloc[[-1]]
        X_scaled = self.trainer.scaler.transform(X)
        prob = float(self.trainer.model.predict_proba(X_scaled)[0, 1])
        confidence = abs(prob - 0.5) * 2   # 0 → no confidence, 1 → full confidence

        if prob >= self.CONFIDENCE_THRESHOLD:
            signal = 1
        elif prob <= (1 - self.CONFIDENCE_THRESHOLD):
            signal = -1
        else:
            signal = 0

        return {
            "signal": signal,
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for every row in `df`.
        Useful for backtesting.
        """
        if not self.is_ready:
            logger.warning("Model not loaded – returning neutral signals")
            df["ai_signal"] = 0
            df["ai_probability"] = 0.5
            return df

        df_feat = self.feature_eng.compute_features(df)
        if df_feat.empty:
            return df

        feature_cols = self.trainer.feature_columns
        for col in feature_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        X = df_feat[feature_cols].fillna(0)
        X_scaled = self.trainer.scaler.transform(X)
        probs = self.trainer.model.predict_proba(X_scaled)[:, 1]

        signals = np.where(probs >= self.CONFIDENCE_THRESHOLD, 1,
                  np.where(probs <= (1 - self.CONFIDENCE_THRESHOLD), -1, 0))

        df_feat["ai_signal"] = signals
        df_feat["ai_probability"] = probs
        return df_feat
