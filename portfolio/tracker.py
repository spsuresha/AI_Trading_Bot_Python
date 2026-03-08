"""
Portfolio tracker.
Maintains the real-time state of capital, positions, PnL, and trade history.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from config.settings import settings
from data_pipeline.storage import DataStorage
from utils.logger import get_logger
from utils.helpers import safe_divide, utc_now

logger = get_logger(__name__)


class Position:
    """Represents a single open position."""

    def __init__(
        self,
        symbol: str,
        direction: int,          # +1 long / -1 short
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        strategy: str = "combined",
        opened_at: Optional[datetime] = None,
    ) -> None:
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy = strategy
        self.opened_at = opened_at or utc_now()
        self.current_price: float = entry_price

    @property
    def unrealised_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity * self.direction

    @property
    def unrealised_pnl_pct(self) -> float:
        return safe_divide(self.unrealised_pnl, self.entry_price * self.quantity) * 100

    @property
    def notional(self) -> float:
        return self.current_price * self.quantity

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "entry_price": round(self.entry_price, 6),
            "current_price": round(self.current_price, 6),
            "quantity": round(self.quantity, 6),
            "notional": round(self.notional, 2),
            "unrealised_pnl": round(self.unrealised_pnl, 4),
            "unrealised_pnl_pct": round(self.unrealised_pnl_pct, 2),
            "stop_loss": round(self.stop_loss, 6),
            "take_profit": round(self.take_profit, 6),
            "strategy": self.strategy,
            "opened_at": self.opened_at.isoformat(),
        }


class PortfolioTracker:
    """
    Tracks the portfolio state across the entire trading session.
    Persists trade history via DataStorage.
    """

    def __init__(
        self,
        initial_capital: Optional[float] = None,
        storage: Optional[DataStorage] = None,
    ) -> None:
        self.initial_capital = initial_capital or settings.risk.initial_capital_inr
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[dict] = []
        self.storage = storage or DataStorage()
        logger.info("PortfolioTracker started | capital=%.2f", self.capital)

    # ------------------------------------------------------------------ #
    #  Position management                                                 #
    # ------------------------------------------------------------------ #

    def open_position(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        strategy: str = "combined",
    ) -> None:
        if symbol in self.positions:
            logger.warning("Position already open for %s – ignoring", symbol)
            return

        pos = Position(symbol, direction, entry_price, quantity, stop_loss, take_profit, strategy)
        self.positions[symbol] = pos
        cost = entry_price * quantity
        self.capital -= cost
        logger.info(
            "Opened %s %s | entry=%.4f | qty=%.6f | capital=%.2f",
            "LONG" if direction == 1 else "SHORT", symbol, entry_price, quantity, self.capital,
        )

    def close_position(
        self, symbol: str, exit_price: float, reason: str = "signal"
    ) -> Optional[dict]:
        if symbol not in self.positions:
            logger.warning("No open position for %s", symbol)
            return None

        pos = self.positions.pop(symbol)
        pnl = (exit_price - pos.entry_price) * pos.quantity * pos.direction
        proceeds = exit_price * pos.quantity + pnl  # return notional + profit
        self.capital += exit_price * pos.quantity   # return cost of position

        trade_record = {
            "symbol": symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "quantity": pos.quantity,
            "pnl": round(pnl, 4),
            "pnl_pct": round(safe_divide(pnl, pos.entry_price * pos.quantity) * 100, 2),
            "strategy": pos.strategy,
            "exit_reason": reason,
            "opened_at": pos.opened_at.isoformat(),
            "closed_at": utc_now().isoformat(),
            "hold_bars": 0,
        }
        self.closed_trades.append(trade_record)

        logger.info(
            "Closed %s %s | exit=%.4f | PnL=%.4f (%.2f%%) | capital=%.2f",
            "LONG" if pos.direction == 1 else "SHORT",
            symbol, exit_price, pnl, trade_record["pnl_pct"], self.capital,
        )
        return trade_record

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all open positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    # ------------------------------------------------------------------ #
    #  State queries                                                       #
    # ------------------------------------------------------------------ #

    @property
    def total_unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self.positions.values())

    @property
    def total_equity(self) -> float:
        return self.capital + sum(p.notional for p in self.positions.values())

    @property
    def total_realised_pnl(self) -> float:
        return sum(t["pnl"] for t in self.closed_trades)

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        return safe_divide(wins, len(self.closed_trades)) * 100

    @property
    def drawdown(self) -> float:
        peak = self.initial_capital + max(
            (sum(t["pnl"] for t in self.closed_trades[:i])
             for i in range(len(self.closed_trades) + 1)),
            default=0.0,
        )
        current = self.total_equity
        return safe_divide(peak - current, peak) * 100

    def get_summary(self) -> dict:
        return {
            "initial_capital": round(self.initial_capital, 2),
            "free_capital": round(self.capital, 2),
            "total_equity": round(self.total_equity, 2),
            "total_return_pct": round(
                safe_divide(self.total_equity - self.initial_capital, self.initial_capital) * 100, 2
            ),
            "realised_pnl": round(self.total_realised_pnl, 4),
            "unrealised_pnl": round(self.total_unrealised_pnl, 4),
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed_trades),
            "win_rate_pct": round(self.win_rate, 2),
        }

    def get_open_positions(self) -> List[dict]:
        return [p.to_dict() for p in self.positions.values()]

    def get_trade_history(self) -> pd.DataFrame:
        if not self.closed_trades:
            return pd.DataFrame()
        df = pd.DataFrame(self.closed_trades)
        df["opened_at"] = pd.to_datetime(df["opened_at"])
        df["closed_at"] = pd.to_datetime(df["closed_at"])
        return df

    def get_pnl_series(self) -> pd.Series:
        """Return cumulative PnL series from closed trades."""
        if not self.closed_trades:
            return pd.Series(dtype=float)
        df = self.get_trade_history().sort_values("closed_at")
        return df["pnl"].cumsum()
