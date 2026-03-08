"""
Model training pipeline.
Trains XGBoost or RandomForest on engineered features and persists to disk.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config.settings import settings
from features.engineer import FeatureEngineer
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Trains, evaluates, and saves a direction-prediction classifier."""

    def __init__(self) -> None:
        self.cfg = settings.model
        self.feature_eng = FeatureEngineer()
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def train(self, df_dict: Dict[str, pd.DataFrame]) -> dict:
        """
        Train on a dictionary of {symbol: ohlcv_df}.
        Concatenates all symbol data, engineers features, and trains.

        Returns a metrics dict.
        """
        logger.info("Starting model training on %d symbols", len(df_dict))
        combined = self._prepare_dataset(df_dict)
        if combined.empty:
            raise ValueError("Empty dataset after feature engineering.")

        X, y = self._split_xy(combined)
        X_train, X_test, y_train, y_test = self._time_split(X, y)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = self._build_model()
        self.model.fit(X_train_scaled, y_train)

        metrics = self._evaluate(X_test_scaled, y_test, X_train_scaled, y_train)
        self.save()
        logger.info("Training complete. Test accuracy: %.4f | AUC: %.4f",
                    metrics["test_accuracy"], metrics.get("test_auc", 0))
        return metrics

    def save(self) -> None:
        model_path = settings.model_dir / self.cfg.model_filename
        scaler_path = settings.model_dir / self.cfg.scaler_filename
        meta_path = settings.model_dir / "feature_columns.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(meta_path, "wb") as f:
            pickle.dump(self.feature_columns, f)

        logger.info("Model saved → %s", model_path)

    def load(self) -> bool:
        model_path = settings.model_dir / self.cfg.model_filename
        scaler_path = settings.model_dir / self.cfg.scaler_filename
        meta_path = settings.model_dir / "feature_columns.pkl"

        if not all(p.exists() for p in (model_path, scaler_path, meta_path)):
            logger.warning("Saved model files not found – train first")
            return False

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(meta_path, "rb") as f:
            self.feature_columns = pickle.load(f)

        logger.info("Model loaded from %s", model_path)
        return True

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _prepare_dataset(self, df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for sym, df_raw in df_dict.items():
            df_feat = self.feature_eng.compute_features(df_raw)
            df_feat = self.feature_eng.add_target(df_feat, self.cfg.target_lookahead)
            df_feat["symbol_hash"] = hash(sym) % 1000   # encode symbol id
            frames.append(df_feat)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=0).sort_index()
        # store feature columns from first frame (excluding target/raw)
        raw_cols = {"open", "high", "low", "close", "volume", "target"}
        self.feature_columns = [c for c in combined.columns if c not in raw_cols]
        return combined

    def _split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df[self.feature_columns]
        y = df["target"]
        return X, y

    def _time_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        split = int(len(X) * self.cfg.train_test_split)
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    def _build_model(self):
        if self.cfg.model_type == "xgboost" and XGBOOST_AVAILABLE:
            params = {k: v for k, v in self.cfg.xgb_params.items()
                      if k != "use_label_encoder"}
            logger.info("Using XGBoost classifier")
            return XGBClassifier(**params, verbosity=0)
        else:
            if self.cfg.model_type == "xgboost" and not XGBOOST_AVAILABLE:
                logger.warning("XGBoost not installed – falling back to RandomForest")
            logger.info("Using RandomForest classifier")
            return RandomForestClassifier(**self.cfg.rf_params)

    def _evaluate(
        self,
        X_test: np.ndarray,
        y_test: pd.Series,
        X_train: np.ndarray,
        y_train: pd.Series,
    ) -> dict:
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "train_accuracy": float(accuracy_score(y_train, self.model.predict(X_train))),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "test_auc": float(roc_auc_score(y_test, y_prob)),
            "classification_report": classification_report(y_test, y_pred),
        }

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            fi = pd.Series(
                self.model.feature_importances_, index=self.feature_columns
            ).sort_values(ascending=False)
            metrics["top_features"] = fi.head(10).to_dict()

        logger.info("Classification report:\n%s", metrics["classification_report"])
        return metrics
