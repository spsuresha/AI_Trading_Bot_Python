"""
Central configuration for the AI Trading Bot.
All parameters are defined here and consumed by all modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ─────────────────────────── paths ───────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "market_data.db"

for _d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────── exchange / api ──────────────────────
@dataclass
class ExchangeConfig:
    exchange_id: str = "binance"          # ccxt exchange id  ("coindcx", "binance", …)
    api_key: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))
    testnet: bool = True                  # True → paper-trading / sandbox
    rate_limit: bool = True


# ─────────────────────── trading universe ────────────────────
@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"
    ])
    timeframe: str = "1h"                 # ccxt timeframe string
    lookback_candles: int = 500           # candles fetched per symbol
    paper_trading: bool = True            # set False for live execution


# ─────────────────────── capital / risk ──────────────────────
@dataclass
class RiskConfig:
    initial_capital_inr: float = 100_000.0   # ₹1 lakh
    risk_per_trade_pct: float = 0.01         # 1 % of capital per trade
    max_daily_loss_pct: float = 0.03         # 3 % daily loss limit
    max_drawdown_pct: float = 0.10           # 10 % portfolio drawdown stop
    stop_loss_atr_mult: float = 2.0          # SL = entry ± 2 × ATR
    take_profit_atr_mult: float = 4.0        # TP = entry ± 4 × ATR
    max_open_positions: int = 4
    min_trade_usdt: float = 10.0             # minimum notional per order


# ─────────────────────── feature engineering ─────────────────
@dataclass
class FeatureConfig:
    rsi_period: int = 14
    ema_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    volume_ma_period: int = 20
    bb_period: int = 20
    bb_std: float = 2.0


# ─────────────────────── ML model ────────────────────────────
@dataclass
class ModelConfig:
    model_type: str = "xgboost"           # "xgboost" | "random_forest"
    target_lookahead: int = 1             # predict 1-candle ahead direction
    train_test_split: float = 0.8
    cv_folds: int = 5
    # XGBoost hyper-params
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    })
    # RandomForest hyper-params
    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 10,
        "random_state": 42,
        "n_jobs": -1,
    })
    model_filename: str = "signal_model.pkl"
    scaler_filename: str = "feature_scaler.pkl"
    retrain_interval_days: int = 7


# ─────────────────────── strategy weights ────────────────────
@dataclass
class StrategyConfig:
    weights: dict = field(default_factory=lambda: {
        "momentum": 0.25,
        "mean_reversion": 0.25,
        "breakout": 0.25,
        "ai_prediction": 0.25,
    })
    signal_threshold: float = 0.5        # combined score to trigger trade
    momentum_lookback: int = 20
    mean_rev_zscore_threshold: float = 2.0
    breakout_lookback: int = 20


# ─────────────────────── reinforcement learning ──────────────
@dataclass
class RLConfig:
    # Training
    total_timesteps: int   = 500_000
    episode_length:  int   = 252      # bars per episode
    # PPO hyper-parameters
    lr:              float = 3e-4
    n_steps:         int   = 2048
    batch_size:      int   = 64
    n_epochs:        int   = 10
    hidden_dim:      int   = 256
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip_range:      float = 0.20
    ent_coef:        float = 0.01
    vf_coef:         float = 0.50
    target_kl:       float = 0.015
    # Strategy integration
    weight:          float = 0.20    # weight in StrategyEngine when RL is enabled
    enabled:         bool  = True    # set False to skip RL in strategy engine


# ─────────────────────── backtesting ─────────────────────────
@dataclass
class BacktestConfig:
    commission_pct: float = 0.001        # 0.1 % taker fee
    slippage_pct: float = 0.0005         # 0.05 % slippage estimate
    initial_capital: float = 100_000.0
    benchmark_symbol: str = "BTC/USDT"


# ─────────────────────── logging ─────────────────────────────
@dataclass
class LogConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_filename: str = str(LOG_DIR / "trading_bot.log")
    max_bytes: int = 10 * 1024 * 1024   # 10 MB
    backup_count: int = 5


# ─────────────────────── composite settings ──────────────────
@dataclass
class Settings:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    logging: LogConfig = field(default_factory=LogConfig)

    # convenience paths
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    model_dir: Path = MODEL_DIR
    log_dir: Path = LOG_DIR
    db_path: Path = DB_PATH

    # ------------------------------------------------------------------ #
    #  GA optimisation helpers                                             #
    # ------------------------------------------------------------------ #

    def apply_params(self, params: dict) -> None:
        """
        Apply a decoded params dict (from GA optimisation) to this Settings
        instance in-place.  Only keys present in `params` are updated so
        it is safe to pass a partial dict.
        """
        # Feature engineering
        if "rsi_period"       in params: self.features.rsi_period       = int(params["rsi_period"])
        if "macd_fast"        in params: self.features.macd_fast        = int(params["macd_fast"])
        if "macd_slow"        in params: self.features.macd_slow        = int(params["macd_slow"])
        if "macd_signal"      in params: self.features.macd_signal      = int(params["macd_signal"])
        if "atr_period"       in params: self.features.atr_period       = int(params["atr_period"])
        if "bb_period"        in params: self.features.bb_period        = int(params["bb_period"])
        if "bb_std"           in params: self.features.bb_std           = float(params["bb_std"])
        if "volume_ma_period" in params: self.features.volume_ma_period = int(params["volume_ma_period"])
        # Strategy
        if "momentum_lookback" in params:
            self.strategy.momentum_lookback = int(params["momentum_lookback"])
        if "zscore_threshold"  in params:
            self.strategy.mean_rev_zscore_threshold = float(params["zscore_threshold"])
        if "breakout_lookback" in params:
            self.strategy.breakout_lookback = int(params["breakout_lookback"])
        if "signal_threshold"  in params:
            self.strategy.signal_threshold  = float(params["signal_threshold"])
        if "weights"           in params:
            self.strategy.weights = dict(params["weights"])
        # Risk management
        if "stop_loss_atr_mult"   in params:
            self.risk.stop_loss_atr_mult   = float(params["stop_loss_atr_mult"])
        if "take_profit_atr_mult" in params:
            self.risk.take_profit_atr_mult = float(params["take_profit_atr_mult"])
        if "risk_per_trade_pct"   in params:
            self.risk.risk_per_trade_pct   = float(params["risk_per_trade_pct"])

    def load_optimized(self) -> bool:
        """
        Load previously saved optimised parameters from
        ``saved_models/optimized_params.json`` and apply them in-place.
        Returns True if the file was found and applied, False otherwise.
        """
        import json
        path = self.model_dir / "optimized_params.json"
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        params = data.get("params", {})
        if not params:
            return False
        self.apply_params(params)
        return True


# Singleton – import `settings` everywhere
settings = Settings()
