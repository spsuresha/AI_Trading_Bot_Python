"""
Professional risk management engine.
Controls position sizing, stop-loss/take-profit, daily loss limits,
and max drawdown protection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from config.settings import settings
from utils.logger import get_logger
from utils.helpers import safe_divide

logger = get_logger(__name__)


@dataclass
class TradeProposal:
    """Input to the risk manager from the strategy engine."""
    symbol: str
    direction: int              # +1 BUY / -1 SELL
    entry_price: float
    atr: float                  # current ATR value
    signal_strength: float      # 0–1


@dataclass
class RiskDecision:
    """Output from the risk manager."""
    approved: bool
    symbol: str
    direction: int
    entry_price: float
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_amount: float = 0.0
    rejection_reason: str = ""


class RiskManager:
    """
    Enforces risk rules before any order is sent:
    1. 1% capital at risk per trade (position sizing via ATR)
    2. ATR-based stop-loss and take-profit levels
    3. Maximum daily loss limit (3%)
    4. Maximum portfolio drawdown protection (10%)
    5. Maximum concurrent open positions
    6. Minimum notional trade value
    """

    def __init__(self, initial_capital: Optional[float] = None) -> None:
        self.cfg = settings.risk
        self.capital: float = initial_capital or self.cfg.initial_capital_inr
        self.peak_capital: float = self.capital
        self.daily_pnl: Dict[date, float] = {}       # date → cumulative pnl
        self.open_positions: Dict[str, dict] = {}     # symbol → position info
        logger.info("RiskManager initialised (capital=%.2f)", self.capital)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(self, proposal: TradeProposal) -> RiskDecision:
        """
        Evaluate a trade proposal and return an approved/rejected decision.
        """
        # --- Pre-trade checks ---
        rejection = self._pre_trade_checks(proposal)
        if rejection:
            return RiskDecision(
                approved=False,
                symbol=proposal.symbol,
                direction=proposal.direction,
                entry_price=proposal.entry_price,
                rejection_reason=rejection,
            )

        # --- Position sizing ---
        risk_amount = self.capital * self.cfg.risk_per_trade_pct
        stop_distance = proposal.atr * self.cfg.stop_loss_atr_mult

        if stop_distance <= 0 or proposal.entry_price <= 0:
            return RiskDecision(
                approved=False,
                symbol=proposal.symbol,
                direction=proposal.direction,
                entry_price=proposal.entry_price,
                rejection_reason="Invalid entry_price or ATR",
            )

        quantity = risk_amount / stop_distance
        notional = quantity * proposal.entry_price

        # Enforce minimum notional
        if notional < self.cfg.min_trade_usdt:
            return RiskDecision(
                approved=False,
                symbol=proposal.symbol,
                direction=proposal.direction,
                entry_price=proposal.entry_price,
                rejection_reason=f"Notional {notional:.2f} below minimum {self.cfg.min_trade_usdt}",
            )

        # --- SL / TP levels ---
        if proposal.direction == 1:   # BUY
            stop_loss = proposal.entry_price - stop_distance
            take_profit = proposal.entry_price + proposal.atr * self.cfg.take_profit_atr_mult
        else:                          # SELL / SHORT
            stop_loss = proposal.entry_price + stop_distance
            take_profit = proposal.entry_price - proposal.atr * self.cfg.take_profit_atr_mult

        decision = RiskDecision(
            approved=True,
            symbol=proposal.symbol,
            direction=proposal.direction,
            entry_price=proposal.entry_price,
            quantity=round(quantity, 6),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_amount=round(risk_amount, 2),
        )
        logger.info(
            "APPROVED %s %s | qty=%.6f | SL=%.4f | TP=%.4f | risk=%.2f",
            "BUY" if proposal.direction == 1 else "SELL",
            proposal.symbol, quantity, stop_loss, take_profit, risk_amount,
        )
        return decision

    def record_trade_open(self, decision: RiskDecision) -> None:
        """Register an opened position."""
        self.open_positions[decision.symbol] = {
            "direction": decision.direction,
            "entry_price": decision.entry_price,
            "quantity": decision.quantity,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "risk_amount": decision.risk_amount,
        }
        logger.info("Position opened: %s", decision.symbol)

    def record_trade_close(self, symbol: str, exit_price: float) -> float:
        """
        Record a closed trade, update capital, and return realised PnL.
        """
        if symbol not in self.open_positions:
            logger.warning("No open position found for %s", symbol)
            return 0.0

        pos = self.open_positions.pop(symbol)
        direction = pos["direction"]
        qty = pos["quantity"]
        entry = pos["entry_price"]

        pnl = (exit_price - entry) * qty * direction
        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

        today = date.today()
        self.daily_pnl[today] = self.daily_pnl.get(today, 0.0) + pnl

        logger.info(
            "Position closed: %s | exit=%.4f | PnL=%.2f | capital=%.2f",
            symbol, exit_price, pnl, self.capital,
        )
        return pnl

    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check whether current price has triggered SL or TP.
        Returns "stop_loss", "take_profit", or None.
        """
        pos = self.open_positions.get(symbol)
        if not pos:
            return None

        direction = pos["direction"]
        sl = pos["stop_loss"]
        tp = pos["take_profit"]

        if direction == 1:   # long
            if current_price <= sl:
                return "stop_loss"
            if current_price >= tp:
                return "take_profit"
        else:                # short
            if current_price >= sl:
                return "stop_loss"
            if current_price <= tp:
                return "take_profit"
        return None

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a fraction (0–1)."""
        return safe_divide(self.peak_capital - self.capital, self.peak_capital)

    @property
    def today_pnl(self) -> float:
        return self.daily_pnl.get(date.today(), 0.0)

    @property
    def today_loss_pct(self) -> float:
        return safe_divide(-self.today_pnl, self.capital)

    def get_summary(self) -> dict:
        return {
            "capital": self.capital,
            "peak_capital": self.peak_capital,
            "drawdown_pct": round(self.current_drawdown * 100, 2),
            "today_pnl": round(self.today_pnl, 2),
            "today_loss_pct": round(self.today_loss_pct * 100, 2),
            "open_positions": len(self.open_positions),
        }

    # ------------------------------------------------------------------ #
    #  Internal checks                                                     #
    # ------------------------------------------------------------------ #

    def _pre_trade_checks(self, proposal: TradeProposal) -> Optional[str]:
        """Return rejection reason string, or None if all checks pass."""

        # Max drawdown protection
        if self.current_drawdown >= self.cfg.max_drawdown_pct:
            return (
                f"Max drawdown reached: {self.current_drawdown*100:.1f}% "
                f">= {self.cfg.max_drawdown_pct*100:.1f}%"
            )

        # Daily loss limit
        if self.today_loss_pct >= self.cfg.max_daily_loss_pct:
            return (
                f"Daily loss limit reached: {self.today_loss_pct*100:.1f}% "
                f">= {self.cfg.max_daily_loss_pct*100:.1f}%"
            )

        # Max open positions
        if len(self.open_positions) >= self.cfg.max_open_positions:
            return (
                f"Max open positions reached: {len(self.open_positions)} "
                f"/ {self.cfg.max_open_positions}"
            )

        # Duplicate position in same symbol
        if proposal.symbol in self.open_positions:
            return f"Already have open position in {proposal.symbol}"

        return None
