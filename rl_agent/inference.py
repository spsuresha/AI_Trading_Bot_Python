"""
Inference wrapper for the trained PPO agent.

Used by:
  • RLStrategy (live signal generation from the strategy engine)
  • Backtesting (batch signal generation over historical data)

The wrapper:
  1. Loads the best saved checkpoint on first access (lazy loading).
  2. Maintains environment state (portfolio + market features) across calls
     so that the agent sees a consistent observation stream.
  3. Exposes a simple `predict(df, symbol)` API that mirrors the ML predictor.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config.settings import settings
from rl_agent.trading_env import TradingEnv, OBS_DIM, ACTION_DIM
from utils.logger import get_logger

logger = get_logger(__name__)

# Action-index → signal mapping
_ACTION_TO_SIGNAL = {0: 0, 1: 1, 2: -1}   # HOLD / BUY / SELL → 0 / +1 / -1
_ACTION_NAMES     = {0: "HOLD", 1: "BUY", 2: "SELL"}

MODEL_PATH      = settings.model_dir / "ppo_agent_best.pt"
FALLBACK_PATH   = settings.model_dir / "ppo_agent.pt"


class PPOInference:
    """
    Single-symbol inference engine.

    Maintains an internal `TradingEnv` positioned at the latest bar and
    queries the policy network for an action on each new candle.

    Parameters
    ----------
    symbol          : trading pair (e.g. "BTC/USDT")
    deterministic   : if True, always take the argmax action (default True)
    model_path      : path to the saved .pt checkpoint
    """

    def __init__(
        self,
        symbol:         str   = "BTC/USDT",
        deterministic:  bool  = True,
        model_path:     Optional[Path] = None,
    ) -> None:
        self.symbol        = symbol
        self.deterministic = deterministic
        self.model_path    = model_path or (MODEL_PATH if MODEL_PATH.exists() else FALLBACK_PATH)

        self._agent   = None
        self._env:    Optional[TradingEnv] = None
        self._loaded  = False

        if self.model_path.exists():
            self._load()
        else:
            logger.warning(
                "PPO model not found at %s – PPO strategy will emit HOLD signals. "
                "Train first: python rl_agent/trainer.py",
                self.model_path,
            )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._agent is not None

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Generate a trading signal from the latest candle.

        Parameters
        ----------
        df : raw OHLCV DataFrame (min ~250 candles for reliable features)

        Returns
        -------
        dict with keys: signal (−1/0/+1), action_name, confidence,
                        portfolio_value, position
        """
        if not self.is_ready:
            return self._neutral()

        # Rebuild env if DataFrame changed significantly (symbol switch / reset)
        if self._env is None or len(df) < 50:
            self._rebuild_env(df)

        obs = self._env._get_obs() if self._env else np.zeros(OBS_DIM, dtype=np.float32)
        action, confidence = self._forward(obs)
        signal = _ACTION_TO_SIGNAL.get(action, 0)

        return {
            "signal":          signal,
            "action":          action,
            "action_name":     _ACTION_NAMES.get(action, "HOLD"),
            "confidence":      round(confidence, 4),
            "portfolio_value": self._env.current_portfolio_value if self._env else 0.0,
            "position":        self._env._position if self._env else 0,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for every bar — used in backtesting.

        Returns the input DataFrame with added columns:
          rl_signal      : int  (−1 / 0 / +1)
          rl_action_name : str
          rl_confidence  : float
        """
        if not self.is_ready:
            df = df.copy()
            df["rl_signal"]      = 0
            df["rl_action_name"] = "HOLD"
            df["rl_confidence"]  = 0.0
            return df

        self._rebuild_env(df)
        if self._env is None:
            df = df.copy()
            df["rl_signal"] = 0
            return df

        signals, actions, confidences = [], [], []

        for i in range(len(self._env._feat_df)):
            self._env._current_idx = i
            obs    = self._env._get_obs()
            action, conf = self._forward(obs)
            sig    = _ACTION_TO_SIGNAL.get(action, 0)

            # Step env to update portfolio state for next observation
            if i < len(self._env._feat_df) - 1:
                try:
                    self._env.step(action)
                except Exception:
                    pass

            signals.append(sig)
            actions.append(_ACTION_NAMES.get(action, "HOLD"))
            confidences.append(conf)

        df = df.copy()
        feat_idx = self._env._feat_df.index if hasattr(self._env._feat_df, 'index') else range(len(signals))

        # Align with original df (feat_df may be shorter due to warmup drops)
        result = df.copy()
        result["rl_signal"]      = 0
        result["rl_action_name"] = "HOLD"
        result["rl_confidence"]  = 0.0

        n = min(len(signals), len(result))
        result.iloc[-n:, result.columns.get_loc("rl_signal")]      = signals[-n:]
        result.iloc[-n:, result.columns.get_loc("rl_action_name")] = actions[-n:]
        result.iloc[-n:, result.columns.get_loc("rl_confidence")]  = confidences[-n:]

        return result

    def update_position(self, action: int) -> None:
        """
        Notify the inference engine of a taken action so its internal
        position state stays in sync with live trading.
        """
        if self._env:
            try:
                self._env.step(action)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available – PPO inference disabled.")
            return

        try:
            from rl_agent.ppo_agent import PPOAgent
            self._agent = PPOAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
            self._agent.load(str(self.model_path))
            self._agent.network.eval()
            self._loaded = True
            logger.info("PPO agent loaded from %s", self.model_path)
        except Exception as exc:
            logger.error("Failed to load PPO agent: %s", exc)
            self._loaded = False

    def _rebuild_env(self, df: pd.DataFrame) -> None:
        """Create a fresh TradingEnv from the given OHLCV DataFrame."""
        try:
            self._env = TradingEnv(
                df              = df,
                symbol          = self.symbol,
                initial_capital = settings.risk.initial_capital_inr,
                episode_length  = len(df),
            )
            # Position the env at the LAST bar for live inference
            self._env._start_idx   = self._env._start_idx
            self._env._current_idx = max(
                self._env._start_idx,
                len(self._env._feat_df) - 1,
            )
        except Exception as exc:
            logger.warning("Could not build TradingEnv for %s: %s", self.symbol, exc)
            self._env = None

    def _forward(self, obs: np.ndarray) -> tuple[int, float]:
        """Run a forward pass and return (action, confidence)."""
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._agent.device)
        with torch.no_grad():
            logits, _ = self._agent.network(obs_t)
            probs      = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        action     = int(np.argmax(probs)) if self.deterministic else \
                     int(np.random.choice(len(probs), p=probs))
        confidence = float(probs[action])
        return action, confidence

    @staticmethod
    def _neutral() -> Dict:
        return {
            "signal": 0, "action": 0,
            "action_name": "HOLD", "confidence": 0.0,
            "portfolio_value": 0.0, "position": 0,
        }


# ─────────────────────── multi-symbol cache ──────────────────

class PPOInferenceCache:
    """
    Caches one `PPOInference` instance per symbol so the same agent
    (and its position state) is reused across strategy calls.
    """

    _instances: Dict[str, PPOInference] = {}

    @classmethod
    def get(cls, symbol: str, **kwargs) -> PPOInference:
        if symbol not in cls._instances:
            cls._instances[symbol] = PPOInference(symbol=symbol, **kwargs)
        return cls._instances[symbol]

    @classmethod
    def reset_all(cls) -> None:
        cls._instances.clear()
