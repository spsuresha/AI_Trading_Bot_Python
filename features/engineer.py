"""
Feature engineering module.
Calculates technical indicators from OHLCV data and returns a
feature matrix ready for machine learning.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Compute technical features from OHLCV DataFrames."""

    def __init__(self) -> None:
        self.cfg = settings.features

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features and return an enriched DataFrame.
        Drops rows that contain NaN in any feature column.
        """
        df = df.copy()
        df = self._add_rsi(df)
        df = self._add_ema(df)
        df = self._add_macd(df)
        df = self._add_atr(df)
        df = self._add_bollinger(df)
        df = self._add_volume_features(df)
        df = self._add_price_features(df)
        df = self._add_candle_patterns(df)

        before = len(df)
        df.dropna(inplace=True)
        logger.debug("Feature engineering: %d → %d rows (dropped %d NaN)", before, len(df), before - len(df))
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return only the engineered feature column names (excludes raw OHLCV)."""
        raw = {"open", "high", "low", "close", "volume"}
        return [c for c in df.columns if c not in raw]

    # ------------------------------------------------------------------ #
    #  Individual indicators                                               #
    # ------------------------------------------------------------------ #

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg.rsi_period
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        return df

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.cfg.ema_periods:
            col = f"ema_{period}"
            df[col] = df["close"].ewm(span=period, adjust=False).mean()
        # EMA trend signals
        if 20 in self.cfg.ema_periods and 50 in self.cfg.ema_periods:
            df["ema_cross_20_50"] = (df["ema_20"] > df["ema_50"]).astype(int)
        if 50 in self.cfg.ema_periods and 200 in self.cfg.ema_periods:
            df["ema_cross_50_200"] = (df["ema_50"] > df["ema_200"]).astype(int)
        # Price relative to EMAs (normalised distance)
        for period in self.cfg.ema_periods:
            df[f"price_to_ema_{period}"] = (df["close"] - df[f"ema_{period}"]) / df[f"ema_{period}"]
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.cfg.macd_fast
        slow = self.cfg.macd_slow
        sig = self.cfg.macd_signal
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=sig, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_bullish"] = (df["macd_hist"] > 0).astype(int)
        df["macd_crossover"] = (
            (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        ).astype(int)
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg.atr_period
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.ewm(com=period - 1, min_periods=period).mean()
        df["atr_pct"] = df["atr"] / df["close"]           # normalised ATR
        return df

    def _add_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg.bb_period
        std_mult = self.cfg.bb_std
        ma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        df["bb_upper"] = ma + std_mult * std
        df["bb_lower"] = ma - std_mult * std
        df["bb_mid"] = ma
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / ma
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ma_period = self.cfg.volume_ma_period
        df["volume_ma"] = df["volume"].rolling(ma_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)
        df["volume_surge"] = (df["volume_ratio"] > 2.0).astype(int)
        # On-Balance Volume
        obv = [0.0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv
        df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()
        df["obv_trend"] = (df["obv"] > df["obv_ema"]).astype(int)
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Returns
        df["return_1"] = df["close"].pct_change(1)
        df["return_3"] = df["close"].pct_change(3)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)
        # Volatility (rolling std of returns)
        df["volatility_5"] = df["return_1"].rolling(5).std()
        df["volatility_20"] = df["return_1"].rolling(20).std()
        # High/Low ratio
        df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]
        # Close position within bar
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
        # Momentum
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
        return df

    def _add_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        body = df["close"] - df["open"]
        candle_range = df["high"] - df["low"] + 1e-9
        df["body_pct"] = body.abs() / candle_range
        df["is_bullish_candle"] = (body > 0).astype(int)
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        df["upper_shadow_pct"] = upper_shadow / candle_range
        df["lower_shadow_pct"] = lower_shadow / candle_range
        # Doji
        df["doji"] = (df["body_pct"] < 0.1).astype(int)
        # Hammer
        df["hammer"] = (
            (lower_shadow > 2 * body.abs()) & (upper_shadow < 0.3 * candle_range)
        ).astype(int)
        return df

    # ------------------------------------------------------------------ #
    #  ML target generation                                                #
    # ------------------------------------------------------------------ #

    def add_target(self, df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
        """
        Add binary classification target:
            1 if close[t+lookahead] > close[t], else 0.
        Drops the last `lookahead` rows (no future data).
        """
        df = df.copy()
        df["target"] = (df["close"].shift(-lookahead) > df["close"]).astype(int)
        df.dropna(subset=["target"], inplace=True)
        return df
