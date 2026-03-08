"""
Fitness evaluation for the genetic algorithm.

Design goals
------------
1. **Self-contained** – `VectorBacktester` takes a `params` dict directly and
   never reads from the global `settings` singleton.  This makes every
   evaluation fully isolated and safe to run in a subprocess.

2. **O(n) signal generation** – all four strategy signals are computed
   vectorially over the pre-computed feature DataFrame, avoiding the O(n²)
   rolling-window loop used in the live backtester.  This gives a ~200×
   speed-up per evaluation and makes the GA practical to run.

3. **Subprocess-safe workers** – module-level `_worker_init` / `_worker_eval`
   functions are picklable so they work with `multiprocessing.Pool` on
   Windows (spawn) and Linux/macOS (fork).

Fitness formula
---------------
    fitness = 0.35 × clamp(Sharpe, -3, 5)
            + 0.25 × clamp(profit_factor − 1, 0, 3)
            + 0.20 × win_rate [0-1]
            − 0.20 × max_drawdown [0-1]

Chromosomes with fewer than MIN_TRADES trades receive a heavy penalty to
discourage over-fitted parameter sets that barely trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import safe_divide

# Minimum trades required for a valid evaluation
MIN_TRADES = 15
# Penalty returned for invalid / crashed evaluations
INVALID_FITNESS = -10.0


# ─────────────────────── feature computation ─────────────────


def compute_features(df: pd.DataFrame, p: Dict) -> pd.DataFrame:
    """
    Compute all required technical features using the parameter dict `p`.
    This is a standalone function so it can be called from any context
    without touching the global settings singleton.
    """
    df = df.copy()
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── RSI ──────────────────────────────────────────────────────
    rsi_period = int(p["rsi_period"])
    delta    = close.diff()
    gain     = delta.clip(lower=0).ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    loss     = (-delta.clip(upper=0)).ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs       = gain / loss.replace(0, np.nan)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # ── EMA (fixed structural periods + optimised MACD) ──────────
    for span in [20, 50, 200]:
        df[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()

    ema_fast = close.ewm(span=int(p["macd_fast"]),   adjust=False).mean()
    ema_slow = close.ewm(span=int(p["macd_slow"]),   adjust=False).mean()
    df["macd"]            = ema_fast - ema_slow
    df["macd_signal_line"]= df["macd"].ewm(span=int(p["macd_signal"]), adjust=False).mean()
    df["macd_hist"]       = df["macd"] - df["macd_signal_line"]

    # ── ATR ───────────────────────────────────────────────────────
    atr_period = int(p["atr_period"])
    hl   = high - low
    hc   = (high - close.shift()).abs()
    lc   = (low  - close.shift()).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=atr_period - 1, min_periods=atr_period).mean()

    # ── Bollinger Bands ───────────────────────────────────────────
    bb_period = int(p["bb_period"])
    bb_std    = float(p["bb_std"])
    bb_ma  = close.rolling(bb_period).mean()
    bb_dev = close.rolling(bb_period).std()
    df["bb_upper"] = bb_ma + bb_std * bb_dev
    df["bb_lower"] = bb_ma - bb_std * bb_dev
    df["bb_pct"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ── Volume ────────────────────────────────────────────────────
    vol_ma = volume.rolling(int(p["volume_ma_period"])).mean()
    df["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

    # ── Rolling Z-score (20-bar) ──────────────────────────────────
    rm20 = close.rolling(20).mean()
    rs20 = close.rolling(20).std()
    df["z_score"] = (close - rm20) / (rs20 + 1e-9)

    # ── Momentum return ───────────────────────────────────────────
    mlb = int(p["momentum_lookback"])
    df["momentum"] = close / close.shift(mlb) - 1.0

    # ── Breakout rolling levels (shift(1) → excludes current bar) ─
    blb = int(p["breakout_lookback"])
    df["roll_high"] = high.rolling(blb).max().shift(1)
    df["roll_low"]  = low.rolling(blb).min().shift(1)

    return df


# ─────────────────────── vectorised signals ──────────────────


def compute_signals(df: pd.DataFrame, p: Dict) -> pd.Series:
    """
    Generate a directional signal (+1 / 0 / -1) for every bar in O(n).

    Each strategy produces a continuous score in [-1, 1]; scores are
    combined with the (normalised) weights from `p["weights"]` and then
    thresholded by `p["signal_threshold"]`.
    """
    weights = p["weights"]
    sig_thr = float(p["signal_threshold"])

    close         = df["close"]
    rsi           = df["rsi"]
    ema_20        = df["ema_20"]
    ema_50        = df["ema_50"]
    macd_hist     = df["macd_hist"]
    momentum      = df["momentum"]
    bb_pct        = df["bb_pct"]
    z_score       = df["z_score"]
    volume_ratio  = df["volume_ratio"].fillna(1.0)
    roll_high     = df["roll_high"]
    roll_low      = df["roll_low"]
    atr           = df["atr"]

    # ── Momentum strategy ─────────────────────────────────────────
    #   Each condition contributes 1/5 to the raw score (max = 1.0)
    mom_buy = (
          (close > ema_20).astype(float)
        + (ema_20 > ema_50).astype(float)
        + ((rsi > 50) & (rsi < 70)).astype(float)
        + (macd_hist > 0).astype(float)
        + (momentum > 0.02).astype(float)
    ) / 5.0

    mom_sell = (
          (close < ema_20).astype(float)
        + (ema_20 < ema_50).astype(float)
        + ((rsi > 30) & (rsi < 50)).astype(float)
        + (macd_hist < 0).astype(float)
        + (momentum < -0.02).astype(float)
    ) / 5.0

    # Signal fires when ≥3/5 conditions are met
    mom_signal = pd.Series(0.0, index=df.index)
    mom_signal[mom_buy  >= 0.60] = mom_buy [mom_buy  >= 0.60]
    mom_signal[mom_sell >= 0.60] = -mom_sell[mom_sell >= 0.60]

    # ── Mean-reversion strategy ───────────────────────────────────
    zt = float(p["zscore_threshold"])

    mr_buy_score = (
          (bb_pct < 0.15).astype(float) * 2.0
        + (bb_pct < 0.25).astype(float) * 1.0
        + (rsi < 35).astype(float)       * 1.5
        + (rsi < 42).astype(float)       * 0.5
        + (z_score < -zt).astype(float)  * 2.0
        + (z_score < -zt * 0.75).astype(float) * 1.0
    ) / 8.0

    mr_sell_score = (
          (bb_pct > 0.85).astype(float) * 2.0
        + (bb_pct > 0.75).astype(float) * 1.0
        + (rsi > 65).astype(float)       * 1.5
        + (rsi > 58).astype(float)       * 0.5
        + (z_score > zt).astype(float)   * 2.0
        + (z_score > zt * 0.75).astype(float) * 1.0
    ) / 8.0

    mr_signal = pd.Series(0.0, index=df.index)
    mr_signal[mr_buy_score  >= 0.50] =  mr_buy_score [mr_buy_score  >= 0.50]
    mr_signal[mr_sell_score >= 0.50] = -mr_sell_score[mr_sell_score >= 0.50]

    # ── Breakout strategy ─────────────────────────────────────────
    vol_surge = volume_ratio > 1.5
    atr_prev  = atr.shift(5).rolling(4).mean()
    atr_exp   = (atr > atr_prev * 1.05).fillna(False)

    bo_buy_raw = (
          (close > roll_high).astype(float) * 0.5
        + vol_surge.astype(float)           * 0.3
        + (volume_ratio > 2.0).astype(float)* 0.1
        + atr_exp.astype(float)             * 0.1
    )
    bo_sell_raw = (
          (close < roll_low).astype(float) * 0.5
        + vol_surge.astype(float)          * 0.3
        + (volume_ratio > 2.0).astype(float)* 0.1
        + atr_exp.astype(float)            * 0.1
    )

    bo_signal = pd.Series(0.0, index=df.index)
    bo_signal[bo_buy_raw  >= 0.5] =  bo_buy_raw [bo_buy_raw  >= 0.5]
    bo_signal[bo_sell_raw >= 0.5] = -bo_sell_raw[bo_sell_raw >= 0.5]

    # ── AI strategy (use pre-computed probabilities if available) ─
    ai_signal = pd.Series(0.0, index=df.index)
    if "ai_probability" in df.columns:
        prob = df["ai_probability"].fillna(0.5)
        ai_signal = (prob - 0.5) * 2.0   # maps 0→-1, 0.5→0, 1→+1
        # Gate: only emit a signal when the model is sufficiently confident
        ai_signal[prob.between(0.45, 0.55)] = 0.0

    # ── Weighted combination ──────────────────────────────────────
    combined = (
          weights["momentum"]       * mom_signal
        + weights["mean_reversion"] * mr_signal
        + weights["breakout"]       * bo_signal
        + weights["ai_prediction"]  * ai_signal
    )

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[combined >=  sig_thr] =  1
    signals[combined <= -sig_thr] = -1

    return signals


# ─────────────────────── simulation loop ─────────────────────


def simulate_trades(
    df: pd.DataFrame,
    signals: pd.Series,
    p: Dict,
    initial_capital: float,
    commission_pct: float = 0.001,
    slippage_pct: float   = 0.0005,
) -> Tuple[List[dict], pd.Series]:
    """
    Simulate trade entries and exits using ATR-based SL/TP.
    Returns (trade_list, equity_series).
    """
    capital     = initial_capital
    equity      = [capital]
    trades: List[dict] = []

    sl_mult  = float(p["stop_loss_atr_mult"])
    tp_mult  = float(p["take_profit_atr_mult"])
    risk_pct = float(p["risk_per_trade_pct"])

    position = None   # {direction, entry, qty, sl, tp, idx}

    for i in range(1, len(df)):
        bar    = df.iloc[i]
        sig    = int(signals.iloc[i])
        close  = float(bar["close"])
        high   = float(bar["high"])
        low    = float(bar["low"])
        atr    = float(bar["atr"]) if not math.isnan(bar.get("atr", float("nan"))) else close * 0.01

        # ── Check existing position SL / TP ──────────────────────
        if position is not None:
            direction = position["direction"]
            exit_price = None
            exit_reason = None

            if direction == 1:    # long
                if low  <= position["sl"]: exit_price, exit_reason = position["sl"], "stop_loss"
                elif high >= position["tp"]: exit_price, exit_reason = position["tp"], "take_profit"
            else:                  # short
                if high >= position["sl"]: exit_price, exit_reason = position["sl"], "stop_loss"
                elif low  <= position["tp"]: exit_price, exit_reason = position["tp"], "take_profit"

            if exit_price is not None:
                qty    = position["qty"]
                entry  = position["entry"]
                gross  = (exit_price - entry) * qty * direction
                comm   = qty * exit_price * commission_pct
                pnl    = gross - comm
                capital += pnl
                trades.append({
                    "entry_idx":   position["idx"],
                    "exit_idx":    i,
                    "direction":   direction,
                    "entry_price": entry,
                    "exit_price":  exit_price,
                    "quantity":    qty,
                    "pnl":         pnl,
                    "pnl_pct":     pnl / max(entry * qty, 1e-9),
                    "exit_reason": exit_reason,
                })
                position = None

        # ── Open new position if flat and signal fires ────────────
        if position is None and sig != 0:
            direction = sig
            entry = close * (1.0 + direction * slippage_pct)
            sl_dist = atr * sl_mult
            if sl_dist <= 0 or entry <= 0:
                equity.append(capital)
                continue

            risk_amount = capital * risk_pct
            qty = risk_amount / sl_dist
            notional = qty * entry
            if notional < initial_capital * 0.001:   # skip dust
                equity.append(capital)
                continue

            capital -= qty * entry * commission_pct   # entry commission
            sl = entry - direction * sl_dist
            tp = entry + direction * atr * tp_mult
            position = {"direction": direction, "entry": entry,
                        "qty": qty, "sl": sl, "tp": tp, "idx": i}

        equity.append(capital)

    # ── Close any open position at last bar ───────────────────────
    if position is not None:
        last_close = float(df.iloc[-1]["close"])
        qty   = position["qty"]
        entry = position["entry"]
        gross = (last_close - entry) * qty * position["direction"]
        comm  = qty * last_close * commission_pct
        pnl   = gross - comm
        capital += pnl
        trades.append({
            "entry_idx":   position["idx"],
            "exit_idx":    len(df) - 1,
            "direction":   position["direction"],
            "entry_price": entry,
            "exit_price":  last_close,
            "quantity":    qty,
            "pnl":         pnl,
            "pnl_pct":     pnl / max(entry * qty, 1e-9),
            "exit_reason": "end_of_data",
        })
        equity[-1] = capital

    eq_series = pd.Series(equity, index=df.index[: len(equity)])
    return trades, eq_series


# ─────────────────────── metrics ─────────────────────────────


def compute_metrics(
    trades: List[dict],
    equity: pd.Series,
    initial_capital: float,
    bars_per_year: int = 365 * 24,   # hourly bars
) -> dict:
    """Compute performance metrics from trade list and equity curve."""
    if equity.empty:
        return _null_metrics()

    final  = float(equity.iloc[-1])
    ret_pct = safe_divide(final - initial_capital, initial_capital) * 100.0

    # CAGR
    n_years = len(equity) / max(bars_per_year, 1)
    cagr = ((final / max(initial_capital, 1e-9)) ** (1.0 / max(n_years, 0.01)) - 1.0) * 100.0

    # Sharpe / Sortino (annualised)
    rets    = equity.pct_change().dropna()
    sharpe  = safe_divide(rets.mean(), rets.std()) * math.sqrt(bars_per_year)
    down    = rets[rets < 0].std()
    sortino = safe_divide(rets.mean(), down) * math.sqrt(bars_per_year)

    # Max drawdown
    roll_max = equity.cummax()
    dd_series = (equity - roll_max) / roll_max.replace(0, np.nan)
    max_dd = abs(float(dd_series.min())) * 100.0

    # Trade stats
    pnls    = [t["pnl"] for t in trades]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]
    win_rate    = safe_divide(len(wins), max(len(pnls), 1)) * 100.0
    pf          = safe_divide(sum(wins), abs(sum(losses)))
    avg_win     = float(np.mean(wins))   if wins   else 0.0
    avg_loss    = float(np.mean(losses)) if losses else 0.0

    return {
        "total_trades":      len(trades),
        "winning_trades":    len(wins),
        "losing_trades":     len(losses),
        "total_return_pct":  round(ret_pct,  3),
        "cagr_pct":          round(cagr,     3),
        "sharpe_ratio":      round(sharpe,   4),
        "sortino_ratio":     round(sortino,  4),
        "max_drawdown_pct":  round(max_dd,   3),
        "win_rate_pct":      round(win_rate, 3),
        "profit_factor":     round(pf,       4),
        "avg_win":           round(avg_win,  4),
        "avg_loss":          round(avg_loss, 4),
        "final_capital":     round(final,    2),
        "initial_capital":   round(initial_capital, 2),
    }


def fitness_from_metrics(m: dict) -> float:
    """
    Compute a scalar fitness score from a metrics dict.

    fitness = 0.35 × clamp(Sharpe, −3, 5)
            + 0.25 × clamp(profit_factor − 1, 0, 3)
            + 0.20 × win_rate [0-1]
            − 0.20 × max_drawdown [0-1]

    Insufficient trades → heavy penalty to deter over-fitted regimes.
    """
    if m["total_trades"] < MIN_TRADES:
        return INVALID_FITNESS + m["total_trades"] * 0.1

    sharpe   = max(-3.0,  min(5.0,  m["sharpe_ratio"]))
    pf_score = max( 0.0,  min(3.0,  m["profit_factor"] - 1.0))
    win_rate = m["win_rate_pct"]  / 100.0
    drawdown = m["max_drawdown_pct"] / 100.0

    return (
          0.35 * sharpe
        + 0.25 * pf_score
        + 0.20 * win_rate
        - 0.20 * drawdown
    )


def _null_metrics() -> dict:
    return {
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "total_return_pct": 0.0, "cagr_pct": 0.0,
        "sharpe_ratio": -10.0, "sortino_ratio": -10.0,
        "max_drawdown_pct": 100.0, "win_rate_pct": 0.0,
        "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "final_capital": 0.0, "initial_capital": 0.0,
    }


# ─────────────────────── evaluator (main process) ────────────


class FitnessEvaluator:
    """
    Evaluates a `Chromosome` against one or more OHLCV DataFrames and
    returns a scalar fitness score.

    This class is used for **sequential** evaluation in the main process
    (no global state is touched).

    For parallel evaluation, use the module-level `_worker_*` functions
    with `multiprocessing.Pool`.
    """

    def __init__(
        self,
        df_dict: Dict[str, pd.DataFrame],
        symbols: List[str],
        initial_capital: float = 100_000.0,
        commission_pct: float  = 0.001,
        slippage_pct: float    = 0.0005,
    ) -> None:
        self.df_dict         = df_dict
        self.symbols         = symbols
        self.initial_capital = initial_capital
        self.commission_pct  = commission_pct
        self.slippage_pct    = slippage_pct

    def evaluate(self, chromosome) -> float:
        """Return scalar fitness for a single chromosome."""
        from optimization.chromosome import Chromosome
        params = chromosome.decode()
        return self._evaluate_params(params)

    def evaluate_with_metrics(self, chromosome) -> Tuple[float, dict]:
        """Return (fitness, combined_metrics_dict) for reporting."""
        from optimization.chromosome import Chromosome
        params = chromosome.decode()
        all_metrics: List[dict] = []

        for sym in self.symbols:
            df_raw = self.df_dict.get(sym)
            if df_raw is None or len(df_raw) < 300:
                continue
            try:
                df_feat = compute_features(df_raw, params)
                df_feat = df_feat.dropna()
                if len(df_feat) < 200:
                    continue
                sigs = compute_signals(df_feat, params)
                trades, equity = simulate_trades(
                    df_feat, sigs, params,
                    self.initial_capital,
                    self.commission_pct,
                    self.slippage_pct,
                )
                m = compute_metrics(trades, equity, self.initial_capital)
                all_metrics.append(m)
            except Exception:
                all_metrics.append(_null_metrics())

        if not all_metrics:
            return INVALID_FITNESS, _null_metrics()

        # Average metrics across symbols
        avg = {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0]
        }
        avg["total_trades"]   = int(sum(m["total_trades"]   for m in all_metrics))
        avg["winning_trades"] = int(sum(m["winning_trades"] for m in all_metrics))
        avg["losing_trades"]  = int(sum(m["losing_trades"]  for m in all_metrics))
        return fitness_from_metrics(avg), avg

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _evaluate_params(self, params: Dict) -> float:
        scores: List[float] = []
        for sym in self.symbols:
            df_raw = self.df_dict.get(sym)
            if df_raw is None or len(df_raw) < 300:
                continue
            try:
                df_feat = compute_features(df_raw, params)
                df_feat = df_feat.dropna()
                if len(df_feat) < 200:
                    continue
                sigs = compute_signals(df_feat, params)
                trades, equity = simulate_trades(
                    df_feat, sigs, params,
                    self.initial_capital,
                    self.commission_pct,
                    self.slippage_pct,
                )
                m = compute_metrics(trades, equity, self.initial_capital)
                scores.append(fitness_from_metrics(m))
            except Exception:
                scores.append(INVALID_FITNESS)

        return float(np.mean(scores)) if scores else INVALID_FITNESS


# ─────────────────────── subprocess workers ──────────────────
# These module-level functions are picklable and used by multiprocessing.Pool.

# Worker-process globals (set once in initializer, then reused per evaluation)
_W_DF_DICT:         Dict[str, pd.DataFrame] = {}
_W_SYMBOLS:         List[str]               = []
_W_INITIAL_CAPITAL: float                   = 100_000.0
_W_COMMISSION:      float                   = 0.001
_W_SLIPPAGE:        float                   = 0.0005


def _worker_init(
    df_dict:         Dict[str, pd.DataFrame],
    symbols:         List[str],
    initial_capital: float,
    commission_pct:  float,
    slippage_pct:    float,
) -> None:
    """Initialise per-process globals (called once per worker process)."""
    global _W_DF_DICT, _W_SYMBOLS, _W_INITIAL_CAPITAL, _W_COMMISSION, _W_SLIPPAGE
    _W_DF_DICT         = df_dict
    _W_SYMBOLS         = symbols
    _W_INITIAL_CAPITAL = initial_capital
    _W_COMMISSION      = commission_pct
    _W_SLIPPAGE        = slippage_pct


def _worker_eval(genes: List[float]) -> float:
    """
    Evaluate a single chromosome (encoded as a list of genes) inside a
    worker process.  Returns a scalar fitness score.
    """
    from optimization.chromosome import Chromosome

    try:
        params = Chromosome(genes).decode()
        scores: List[float] = []

        for sym in _W_SYMBOLS:
            df_raw = _W_DF_DICT.get(sym)
            if df_raw is None or len(df_raw) < 300:
                continue
            try:
                df_feat = compute_features(df_raw, params)
                df_feat = df_feat.dropna()
                if len(df_feat) < 200:
                    continue
                sigs = compute_signals(df_feat, params)
                trades, equity = simulate_trades(
                    df_feat, sigs, params,
                    _W_INITIAL_CAPITAL, _W_COMMISSION, _W_SLIPPAGE,
                )
                m = compute_metrics(trades, equity, _W_INITIAL_CAPITAL)
                scores.append(fitness_from_metrics(m))
            except Exception:
                scores.append(INVALID_FITNESS)

        return float(np.mean(scores)) if scores else INVALID_FITNESS

    except Exception:
        return INVALID_FITNESS
