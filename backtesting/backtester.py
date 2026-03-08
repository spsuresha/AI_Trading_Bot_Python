"""
Vectorised backtesting engine.
Replays historical data against a strategy engine (or pre-computed signals)
and computes professional performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from features.engineer import FeatureEngineer
from strategies.engine import StrategyEngine, SignalDirection
from utils.logger import get_logger
from utils.helpers import safe_divide

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    best_trade_pct: float
    worst_trade_pct: float
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy_name,
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "cagr_pct": round(self.cagr_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "profit_factor": round(self.profit_factor, 3),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
        }


class Backtester:
    """
    Vectorised backtest on a single symbol.

    The strategy engine generates signals on a rolling basis;
    entry/exit logic uses ATR-based SL/TP with commission & slippage.
    """

    def __init__(
        self,
        strategy_engine: Optional[StrategyEngine] = None,
        initial_capital: Optional[float] = None,
    ) -> None:
        self.engine = strategy_engine or StrategyEngine()
        self.cfg = settings.backtest
        self.risk_cfg = settings.risk
        self.initial_capital = initial_capital or self.cfg.initial_capital
        self.feature_eng = FeatureEngineer()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self, df_raw: pd.DataFrame, symbol: str) -> BacktestResult:
        """
        Run a full backtest on `df_raw` OHLCV data.
        Uses a walk-forward approach: at each bar, signals are generated
        from all data up to (but not including) that bar.
        """
        logger.info("Starting backtest for %s (%d candles)", symbol, len(df_raw))

        df = self.feature_eng.compute_features(df_raw.copy())
        if df.empty or len(df) < 100:
            raise ValueError(f"Insufficient data for backtesting {symbol}")

        # Pre-compute signals for entire dataset (vectorised)
        signals = self._vectorised_signals(df, symbol)

        # Simulate trades
        trades, equity = self._simulate(df, signals)

        result = self._compute_metrics(symbol, trades, equity)
        logger.info("Backtest complete: %s | return=%.2f%% | sharpe=%.3f | dd=%.2f%%",
                    symbol, result.total_return_pct, result.sharpe_ratio, result.max_drawdown_pct)
        return result

    def run_all_symbols(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, BacktestResult]:
        results = {}
        for sym, df in data.items():
            try:
                results[sym] = self.run(df, sym)
            except Exception as exc:
                logger.error("Backtest failed for %s: %s", sym, exc)
        return results

    # ------------------------------------------------------------------ #
    #  Signal generation (vectorised)                                      #
    # ------------------------------------------------------------------ #

    def _vectorised_signals(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate a signal (+1/-1/0) for every bar using rolling window."""
        signals = pd.Series(0, index=df.index, dtype=int)
        min_window = 200   # minimum bars before generating signals

        for i in range(min_window, len(df)):
            window = df.iloc[:i]
            try:
                combined = self.engine.process(window, symbol)
                signals.iloc[i] = int(combined.direction)
            except Exception:
                signals.iloc[i] = 0

        return signals

    # ------------------------------------------------------------------ #
    #  Trade simulation                                                    #
    # ------------------------------------------------------------------ #

    def _simulate(
        self, df: pd.DataFrame, signals: pd.Series
    ) -> tuple[List[dict], pd.Series]:
        """
        Simulate entry/exit logic with ATR-based SL/TP, commission, and slippage.
        Returns (trade_list, equity_series).
        """
        capital = self.initial_capital
        equity = [capital]
        trades: List[dict] = []

        position = None   # None | {"direction", "entry_price", "qty", "sl", "tp", "entry_idx"}

        for i in range(1, len(df)):
            bar = df.iloc[i]
            prev_signal = signals.iloc[i]
            close = float(bar["close"])
            atr = float(bar.get("atr", close * 0.01))

            # Check SL/TP if in position
            if position is not None:
                exit_reason, exit_price = self._check_exit(bar, position)
                if exit_reason:
                    pnl = self._close_trade(position, exit_price, capital)
                    capital += pnl
                    trades.append({
                        "entry_idx": position["entry_idx"],
                        "exit_idx": i,
                        "symbol": df.index[i],
                        "direction": position["direction"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "quantity": position["qty"],
                        "pnl": pnl,
                        "pnl_pct": pnl / (position["entry_price"] * position["qty"]),
                        "exit_reason": exit_reason,
                    })
                    position = None

            # Open new position (only if flat)
            if position is None and prev_signal != 0:
                direction = prev_signal
                entry = close * (1 + direction * self.cfg.slippage_pct)
                risk_amount = capital * self.risk_cfg.risk_per_trade_pct
                sl_dist = atr * self.risk_cfg.stop_loss_atr_mult
                if sl_dist <= 0:
                    equity.append(capital)
                    continue

                qty = risk_amount / sl_dist
                if qty * entry < self.cfg.initial_capital * 0.001:
                    equity.append(capital)
                    continue

                # Commission on entry
                commission = qty * entry * self.cfg.commission_pct
                capital -= commission

                sl = entry - direction * sl_dist
                tp = entry + direction * atr * self.risk_cfg.take_profit_atr_mult
                position = {
                    "direction": direction,
                    "entry_price": entry,
                    "qty": qty,
                    "sl": sl,
                    "tp": tp,
                    "entry_idx": i,
                }

            equity.append(capital)

        # Close any open position at last bar
        if position is not None:
            last_close = float(df.iloc[-1]["close"])
            pnl = self._close_trade(position, last_close, capital)
            capital += pnl
            trades.append({
                "entry_idx": position["entry_idx"],
                "exit_idx": len(df) - 1,
                "direction": position["direction"],
                "entry_price": position["entry_price"],
                "exit_price": last_close,
                "quantity": position["qty"],
                "pnl": pnl,
                "pnl_pct": pnl / (position["entry_price"] * position["qty"]),
                "exit_reason": "end_of_data",
            })
            equity[-1] = capital

        equity_series = pd.Series(equity, index=df.index[:len(equity)])
        return trades, equity_series

    def _check_exit(self, bar: pd.Series, position: dict) -> tuple[Optional[str], float]:
        high = float(bar["high"])
        low = float(bar["low"])
        direction = position["direction"]
        sl = position["sl"]
        tp = position["tp"]

        if direction == 1:   # long
            if low <= sl:
                return "stop_loss", sl
            if high >= tp:
                return "take_profit", tp
        else:                # short
            if high >= sl:
                return "stop_loss", sl
            if low <= tp:
                return "take_profit", tp
        return None, 0.0

    def _close_trade(self, position: dict, exit_price: float, capital: float) -> float:
        direction = position["direction"]
        qty = position["qty"]
        entry = position["entry_price"]
        gross_pnl = (exit_price - entry) * qty * direction
        commission = qty * exit_price * self.cfg.commission_pct
        return gross_pnl - commission

    # ------------------------------------------------------------------ #
    #  Performance metrics                                                 #
    # ------------------------------------------------------------------ #

    def _compute_metrics(
        self,
        symbol: str,
        trades: List[dict],
        equity: pd.Series,
    ) -> BacktestResult:
        initial = self.initial_capital
        final = float(equity.iloc[-1])
        total_return = safe_divide(final - initial, initial) * 100

        # CAGR
        n_years = len(equity) / (365 * 24)   # assume hourly bars
        cagr = ((final / initial) ** (1 / max(n_years, 0.01)) - 1) * 100

        # Returns series
        ret = equity.pct_change().dropna()

        # Sharpe (annualised, assuming hourly bars → 365*24 periods/year)
        periods_year = 365 * 24
        sharpe = safe_divide(ret.mean(), ret.std()) * np.sqrt(periods_year)

        # Sortino
        downside = ret[ret < 0].std()
        sortino = safe_divide(ret.mean(), downside) * np.sqrt(periods_year)

        # Max drawdown
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = float(drawdown.min()) * 100

        # Trade stats
        pnl_list = [t["pnl"] for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        win_rate = safe_divide(len(wins), len(pnl_list)) * 100
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        profit_factor = safe_divide(sum(wins), abs(sum(losses)))
        pnl_pcts = [t.get("pnl_pct", 0) for t in trades]

        trade_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return BacktestResult(
            symbol=symbol,
            strategy_name="combined",
            initial_capital=initial,
            final_capital=final,
            total_return_pct=total_return,
            cagr_pct=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=abs(max_dd),
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade_pct=max(pnl_pcts) * 100 if pnl_pcts else 0.0,
            worst_trade_pct=min(pnl_pcts) * 100 if pnl_pcts else 0.0,
            equity_curve=equity,
            trade_log=trade_df,
        )
