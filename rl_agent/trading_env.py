"""
Custom Gymnasium-compatible trading environment for PPO training.

Observation space  (24-dimensional float32 vector)
──────────────────────────────────────────────────
Idx  Feature                   Range after normalisation
  0  RSI / 100                 [0, 1]
  1  (close/EMA-20) − 1        clipped ±0.10, normalised
  2  (close/EMA-50) − 1        clipped ±0.20, normalised
  3  (close/EMA-200) − 1       clipped ±0.30, normalised
  4  MACD-hist / ATR            clipped ±2,   normalised
  5  ATR / close                clipped  0.10, normalised
  6  BB-pct                     [0, 1]
  7  volume_ratio / 4           clipped [0, 1]
  8  Z-score / 3                clipped ±1
  9  10-bar momentum            clipped ±0.20, normalised
 10  20-bar momentum            clipped ±0.30, normalised
 11  1-bar return               clipped ±0.05, normalised
 12  3-bar return               clipped ±0.10, normalised
 13  5-bar return               clipped ±0.15, normalised
 14  20-bar volatility          clipped  0.05, normalised
 15  EMA-20 > EMA-50  (binary)  {0, 1}
 16  MACD-hist > 0    (binary)  {0, 1}
 17  current position           {−1, 0, 1}
 18  unrealised PnL %           clipped ±0.20, normalised
 19  bars in trade / 100        clipped [0, 1]
 20  portfolio return vs start  clipped ±0.50, normalised
 21  current drawdown           clipped [0, 1]
 22  hour of day / 23           [0, 1]  (0.5 if no timestamp)
 23  day of week / 6            [0, 1]

Action space (Discrete 3)
──────────────────────────
  0  HOLD   – maintain current position
  1  BUY    – open / switch to long
  2  SELL   – open / switch to short

Reward
──────
  r_t = log(V_t / V_{t-1}) × 100   (per-step log portfolio return, scaled)
      − transaction_cost             (charged when position changes)
      + sharpe_bonus                 (at episode termination only)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────── constants ───────────────────────────
OBS_DIM        = 24
ACTION_DIM     = 3          # HOLD / BUY / SELL
MIN_WARMUP     = 210        # bars needed before the first valid observation
TRANSACTION_COST = 0.001    # 0.1 % round-trip cost per trade


# ─────────────────────── feature helper ──────────────────────

def _build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute all features needed for the observation vector.
    Called once in __init__; the env then indexes by row position.
    """
    d       = df.copy()
    close   = d["close"]
    high    = d["high"]
    low     = d["low"]
    volume  = d["volume"]

    # RSI (14)
    delta  = close.diff()
    gain   = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # EMAs
    for span in (20, 50, 200):
        d[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    # ATR (14)
    hl  = high - low
    hc  = (high - close.shift()).abs()
    lc  = (low  - close.shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr"] = tr.ewm(com=13, min_periods=14).mean()

    # Bollinger Bands (20, 2σ)
    bb_ma  = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    d["bb_pct"] = (close - (bb_ma - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # Volume ratio
    vol_ma = volume.rolling(20).mean()
    d["vol_ratio"] = volume / vol_ma.replace(0, np.nan)

    # Rolling Z-score (20)
    rm20 = close.rolling(20).mean()
    rs20 = close.rolling(20).std()
    d["z_score"] = (close - rm20) / (rs20 + 1e-9)

    # Momentum / returns
    d["mom_10"]  = close / close.shift(10)  - 1
    d["mom_20"]  = close / close.shift(20)  - 1
    d["ret_1"]   = close.pct_change(1)
    d["ret_3"]   = close.pct_change(3)
    d["ret_5"]   = close.pct_change(5)
    d["vol_20"]  = d["ret_1"].rolling(20).std()

    d.dropna(inplace=True)
    d.reset_index(drop=False, inplace=True)   # keep timestamp as column
    return d


def _clip_norm(val: float, lo: float, hi: float) -> float:
    """Clip to [lo, hi] then normalise to [−1, 1] (or [0,1] if lo=0)."""
    val = float(np.clip(val, lo, hi))
    rng = hi - lo
    if rng == 0:
        return 0.0
    mid = (lo + hi) / 2.0
    return (val - mid) / (rng / 2.0)


# ─────────────────────── environment ─────────────────────────

class TradingEnv:
    """
    A single-asset trading environment.

    Compatible with the Gymnasium step API:
        obs, info          = env.reset()
        obs, r, done, trunc, info = env.step(action)

    If the `gymnasium` package is available the class also inherits from
    `gym.Env` so it can be used directly with Gymnasium-based tooling.
    """

    # Gymnasium metadata (ignored if gym not installed)
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str               = "ASSET",
        initial_capital: float    = 100_000.0,
        episode_length: int       = 252,
        commission: float         = TRANSACTION_COST,
        short_selling: bool       = True,
        sharpe_bonus_coef: float  = 0.05,
        seed: Optional[int]       = None,
    ) -> None:
        self.symbol          = symbol
        self.initial_capital = initial_capital
        self.episode_length  = episode_length
        self.commission      = commission
        self.short_selling   = short_selling
        self.sharpe_bonus    = sharpe_bonus_coef
        self._rng            = np.random.default_rng(seed)

        # Pre-compute features once
        self._feat_df = _build_feature_df(df)
        if len(self._feat_df) < MIN_WARMUP + episode_length:
            raise ValueError(
                f"DataFrame too short for RL env: need {MIN_WARMUP + episode_length} "
                f"rows after feature computation, got {len(self._feat_df)}"
            )

        # Gymnasium spaces (if available)
        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(ACTION_DIM)

        # Episode state (initialised in reset)
        self._start_idx:     int   = MIN_WARMUP
        self._current_idx:   int   = MIN_WARMUP
        self._position:      int   = 0          # −1 short | 0 flat | +1 long
        self._entry_price:   float = 0.0
        self._bars_in_trade: int   = 0
        self._capital:       float = initial_capital
        self._portfolio_val: float = initial_capital
        self._peak_val:      float = initial_capital
        self._step_returns:  list  = []
        self._done:          bool  = False

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        max_start = len(self._feat_df) - self.episode_length - 1
        self._start_idx   = int(self._rng.integers(MIN_WARMUP, max(MIN_WARMUP + 1, max_start)))
        self._current_idx = self._start_idx

        self._position      = 0
        self._entry_price   = 0.0
        self._bars_in_trade = 0
        self._capital       = self.initial_capital
        self._portfolio_val = self.initial_capital
        self._peak_val      = self.initial_capital
        self._step_returns  = []
        self._done          = False

        obs = self._get_obs()
        return obs, {"symbol": self.symbol, "start_idx": self._start_idx}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert not self._done, "Call reset() before stepping a done episode."
        assert action in (0, 1, 2), f"Invalid action {action}"

        prev_val = self._portfolio_val
        prev_pos = self._position

        # ── Execute action ────────────────────────────────────────
        row      = self._feat_df.iloc[self._current_idx]
        close    = float(row["close"])
        reward   = 0.0

        new_pos = self._action_to_position(action)
        if not self.short_selling and new_pos == -1:
            new_pos = 0   # disallow shorts

        position_changed = new_pos != prev_pos

        if position_changed:
            # Close previous position
            if prev_pos != 0 and self._entry_price > 0:
                exit_return = (close - self._entry_price) / self._entry_price * prev_pos
                self._capital *= (1.0 + exit_return)
            # Apply transaction cost
            self._capital *= (1.0 - self.commission)
            # Open new position
            self._position    = new_pos
            self._entry_price = close if new_pos != 0 else 0.0
            self._bars_in_trade = 0
        else:
            self._bars_in_trade += 1

        # ── Portfolio value (mark-to-market) ─────────────────────
        if self._position != 0 and self._entry_price > 0:
            unreal_ret = (close - self._entry_price) / self._entry_price * self._position
            self._portfolio_val = self._capital * (1.0 + unreal_ret)
        else:
            self._portfolio_val = self._capital

        # ── Reward = scaled log portfolio return ──────────────────
        safe_prev = max(prev_val, 1e-9)
        log_ret   = math.log(max(self._portfolio_val, 1e-9) / safe_prev)
        step_ret  = (self._portfolio_val - prev_val) / safe_prev
        self._step_returns.append(step_ret)
        reward    = log_ret * 100.0   # scale for gradient stability

        # Update peak (for drawdown tracking)
        self._peak_val = max(self._peak_val, self._portfolio_val)

        # ── Advance timestep ──────────────────────────────────────
        self._current_idx += 1
        steps_taken = self._current_idx - self._start_idx
        terminated  = steps_taken >= self.episode_length
        truncated   = False

        # ── Episode-end Sharpe bonus ──────────────────────────────
        if terminated and len(self._step_returns) >= 10:
            rets  = np.array(self._step_returns, dtype=float)
            sharpe = np.mean(rets) / (np.std(rets) + 1e-9) * math.sqrt(252)
            reward += self.sharpe_bonus * float(np.clip(sharpe, -3.0, 5.0))

        self._done = terminated

        obs  = self._get_obs()
        info = {
            "portfolio_value":   self._portfolio_val,
            "position":          self._position,
            "step_return":       step_ret,
            "drawdown":          (self._peak_val - self._portfolio_val) / max(self._peak_val, 1e-9),
            "bars_in_trade":     self._bars_in_trade,
            "position_changed":  position_changed,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        row   = self._feat_df.iloc[self._current_idx]
        ts    = row.get("timestamp", self._current_idx)
        msg   = (
            f"[{ts}] pos={self._position:+d} "
            f"close={row['close']:.4f} "
            f"portfolio={self._portfolio_val:.2f} "
            f"drawdown={((self._peak_val - self._portfolio_val)/max(self._peak_val,1e-9))*100:.1f}%"
        )
        if mode == "human":
            print(msg)
        return msg

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Observation builder                                                 #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        idx = min(self._current_idx, len(self._feat_df) - 1)
        row = self._feat_df.iloc[idx]
        c   = float(row["close"])

        def _safe(col: str, default: float = 0.0) -> float:
            v = row.get(col, default)
            return float(v) if not (v != v) else default   # NaN check

        rsi      = _safe("rsi", 50.0)
        ema20    = _safe("ema_20", c)
        ema50    = _safe("ema_50", c)
        ema200   = _safe("ema_200", c)
        mhist    = _safe("macd_hist", 0.0)
        atr      = _safe("atr", c * 0.01)
        bb_pct   = _safe("bb_pct", 0.5)
        vol_r    = _safe("vol_ratio", 1.0)
        z        = _safe("z_score", 0.0)
        m10      = _safe("mom_10", 0.0)
        m20      = _safe("mom_20", 0.0)
        r1       = _safe("ret_1", 0.0)
        r3       = _safe("ret_3", 0.0)
        r5       = _safe("ret_5", 0.0)
        vol20    = _safe("vol_20", 0.01)

        unreal_pnl = 0.0
        if self._position != 0 and self._entry_price > 0:
            unreal_pnl = (c - self._entry_price) / self._entry_price * self._position

        portfolio_ret = (self._portfolio_val - self.initial_capital) / self.initial_capital
        drawdown = (self._peak_val - self._portfolio_val) / max(self._peak_val, 1e-9)

        # Time features
        ts = row.get("timestamp", None)
        try:
            hour     = float(pd.Timestamp(ts).hour)   / 23.0
            dow      = float(pd.Timestamp(ts).dayofweek) / 6.0
        except Exception:
            hour, dow = 0.5, 0.5

        obs = np.array([
            # Market features
            np.clip(rsi / 100.0, 0.0, 1.0),
            _clip_norm((c / max(ema20,  1e-9) - 1.0), -0.10, 0.10),
            _clip_norm((c / max(ema50,  1e-9) - 1.0), -0.20, 0.20),
            _clip_norm((c / max(ema200, 1e-9) - 1.0), -0.30, 0.30),
            _clip_norm(mhist / max(atr, 1e-9),         -2.0,   2.0),
            float(np.clip(atr / max(c, 1e-9), 0.0, 0.10)) / 0.10,
            float(np.clip(bb_pct, 0.0, 1.0)),
            float(np.clip(vol_r / 4.0, 0.0, 1.0)),
            _clip_norm(z / 3.0, -1.0, 1.0),
            _clip_norm(m10,  -0.20, 0.20),
            _clip_norm(m20,  -0.30, 0.30),
            _clip_norm(r1,   -0.05, 0.05),
            _clip_norm(r3,   -0.10, 0.10),
            _clip_norm(r5,   -0.15, 0.15),
            float(np.clip(vol20 / 0.05, 0.0, 1.0)),
            float(ema20 > ema50),
            float(mhist > 0.0),
            # Portfolio state
            float(self._position),                                              # −1/0/+1
            _clip_norm(unreal_pnl,    -0.20, 0.20),
            float(np.clip(self._bars_in_trade / 100.0, 0.0, 1.0)),
            _clip_norm(portfolio_ret, -0.50, 0.50),
            float(np.clip(drawdown, 0.0, 1.0)),
            # Time
            hour,
            dow,
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _action_to_position(action: int) -> int:
        """Map discrete action → position direction."""
        return {0: None, 1: 1, 2: -1}.get(action, None)  # type: ignore

    def _action_to_position(self, action: int) -> int:
        if action == 1:
            return 1
        elif action == 2:
            return -1
        else:
            return self._position   # HOLD: keep current position

    @property
    def current_portfolio_value(self) -> float:
        return self._portfolio_val

    @property
    def total_return_pct(self) -> float:
        return (self._portfolio_val / self.initial_capital - 1.0) * 100.0


# ─────────────────────── multi-asset wrapper ─────────────────

class MultiAssetEnv:
    """
    Wraps multiple `TradingEnv` instances (one per symbol) and randomly
    selects one environment per episode.  Used during PPO training to
    produce a more generalised policy.
    """

    def __init__(
        self,
        df_dict: Dict[str, pd.DataFrame],
        **env_kwargs,
    ) -> None:
        self.envs: Dict[str, TradingEnv] = {}
        for sym, df in df_dict.items():
            try:
                self.envs[sym] = TradingEnv(df, symbol=sym, **env_kwargs)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", sym, exc)

        if not self.envs:
            raise ValueError("No valid environments created – check data length.")

        self._symbols       = list(self.envs.keys())
        self._active_env:   Optional[TradingEnv] = None
        self._active_sym:   str = self._symbols[0]
        self._rng           = np.random.default_rng(env_kwargs.get("seed"))

        # Expose spaces from first env
        first = next(iter(self.envs.values()))
        if GYM_AVAILABLE:
            self.observation_space = first.observation_space
            self.action_space      = first.action_space

        logger.info("MultiAssetEnv: %d symbols — %s", len(self.envs), self._symbols)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self._active_sym = self._rng.choice(self._symbols)
        self._active_env = self.envs[self._active_sym]
        return self._active_env.reset(**kwargs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._active_env is not None, "Call reset() first."
        return self._active_env.step(action)

    def render(self, **kwargs) -> Optional[str]:
        return self._active_env.render(**kwargs) if self._active_env else None

    def close(self) -> None:
        for env in self.envs.values():
            env.close()
