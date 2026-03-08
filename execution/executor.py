"""
Order execution engine.
Supports both paper trading (simulation) and live exchange execution.
Connects to the exchange via ccxt, logs every order to SQLite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import ccxt

from config.settings import settings
from data_pipeline.storage import DataStorage
from utils.logger import get_logger
from utils.helpers import round_price, round_qty, utc_now

logger = get_logger(__name__)


@dataclass
class Order:
    """Describes an order to be placed."""
    symbol: str
    side: str                   # "buy" | "sell"
    quantity: float
    order_type: str = "market"  # "market" | "limit"
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = "combined"
    metadata: dict = field(default_factory=dict)


@dataclass
class OrderResult:
    """Result after order placement / simulation."""
    success: bool
    order_id: str
    symbol: str
    side: str
    filled_price: float
    filled_qty: float
    commission: float
    timestamp: datetime
    paper: bool
    raw: dict = field(default_factory=dict)
    error: str = ""

    @property
    def notional(self) -> float:
        return self.filled_price * self.filled_qty


class OrderExecutor:
    """
    Places orders via ccxt or simulates them in paper-trading mode.
    Every order result is persisted to the trade log in SQLite.
    """

    def __init__(
        self,
        storage: Optional[DataStorage] = None,
    ) -> None:
        self.cfg_exchange = settings.exchange
        self.cfg_trading = settings.trading
        self.storage = storage or DataStorage()
        self.paper_mode = self.cfg_trading.paper_trading

        if not self.paper_mode:
            self._exchange = self._init_exchange()
        else:
            self._exchange = None

        logger.info(
            "OrderExecutor initialised [%s mode]",
            "PAPER" if self.paper_mode else "LIVE",
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def execute(self, order: Order, current_price: Optional[float] = None) -> OrderResult:
        """
        Execute an order.
        In paper mode, simulate fill at `current_price` (or order.price).
        In live mode, send a real market/limit order to the exchange.
        """
        if self.paper_mode:
            return self._paper_execute(order, current_price)
        return self._live_execute(order)

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self.paper_mode:
            logger.info("PAPER: Cancel order %s (no-op)", order_id)
            return True
        try:
            self._exchange.cancel_order(order_id, symbol)
            logger.info("Cancelled order %s for %s", order_id, symbol)
            return True
        except Exception as exc:
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False

    def get_open_orders(self, symbol: str) -> list:
        if self.paper_mode:
            return []
        try:
            return self._exchange.fetch_open_orders(symbol)
        except Exception as exc:
            logger.error("Failed to fetch open orders: %s", exc)
            return []

    def get_balance(self) -> dict:
        if self.paper_mode:
            return {"USDT": {"free": 0, "used": 0, "total": 0}}
        try:
            return self._exchange.fetch_balance()
        except Exception as exc:
            logger.error("Failed to fetch balance: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Paper trading                                                       #
    # ------------------------------------------------------------------ #

    def _paper_execute(self, order: Order, current_price: Optional[float]) -> OrderResult:
        slippage = settings.backtest.slippage_pct
        commission_rate = settings.backtest.commission_pct

        fill_price = current_price or order.price or 0.0
        if fill_price <= 0:
            return OrderResult(
                success=False,
                order_id="paper_error",
                symbol=order.symbol,
                side=order.side,
                filled_price=0.0,
                filled_qty=0.0,
                commission=0.0,
                timestamp=utc_now(),
                paper=True,
                error="No price available for paper fill",
            )

        # Apply slippage
        if order.side == "buy":
            fill_price *= (1 + slippage)
        else:
            fill_price *= (1 - slippage)

        commission = fill_price * order.quantity * commission_rate
        order_id = f"paper_{int(time.time() * 1000)}"

        result = OrderResult(
            success=True,
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            filled_price=round(fill_price, 6),
            filled_qty=order.quantity,
            commission=round(commission, 6),
            timestamp=utc_now(),
            paper=True,
        )

        self._log_order(result, order)
        logger.info(
            "PAPER %s %s | qty=%.6f | price=%.4f | commission=%.4f",
            order.side.upper(), order.symbol, order.quantity, fill_price, commission,
        )
        return result

    # ------------------------------------------------------------------ #
    #  Live execution                                                      #
    # ------------------------------------------------------------------ #

    def _live_execute(self, order: Order) -> OrderResult:
        try:
            if order.order_type == "market":
                raw = self._exchange.create_market_order(
                    order.symbol,
                    order.side,
                    order.quantity,
                )
            else:
                if not order.price:
                    raise ValueError("Limit order requires a price")
                raw = self._exchange.create_limit_order(
                    order.symbol,
                    order.side,
                    order.quantity,
                    order.price,
                )

            filled_price = float(raw.get("average") or raw.get("price") or 0)
            filled_qty = float(raw.get("filled") or order.quantity)
            commission = float(
                raw.get("fee", {}).get("cost", filled_price * filled_qty * 0.001)
            )

            result = OrderResult(
                success=True,
                order_id=str(raw.get("id", "")),
                symbol=order.symbol,
                side=order.side,
                filled_price=filled_price,
                filled_qty=filled_qty,
                commission=commission,
                timestamp=utc_now(),
                paper=False,
                raw=raw,
            )
            self._log_order(result, order)
            logger.info(
                "LIVE %s %s | qty=%.6f | price=%.4f",
                order.side.upper(), order.symbol, filled_qty, filled_price,
            )
            return result

        except ccxt.InsufficientFunds as exc:
            logger.error("Insufficient funds for %s %s: %s", order.side, order.symbol, exc)
            return self._error_result(order, str(exc))
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error for %s %s: %s", order.side, order.symbol, exc)
            return self._error_result(order, str(exc))
        except Exception as exc:
            logger.error("Unexpected error placing order: %s", exc, exc_info=True)
            return self._error_result(order, str(exc))

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _init_exchange(self) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, self.cfg_exchange.exchange_id)
        exchange = exchange_class({
            "apiKey": self.cfg_exchange.api_key,
            "secret": self.cfg_exchange.api_secret,
            "enableRateLimit": self.cfg_exchange.rate_limit,
        })
        if self.cfg_exchange.testnet and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)
        return exchange

    def _log_order(self, result: OrderResult, order: Order) -> None:
        trade_record = {
            "timestamp": result.timestamp.isoformat(),
            "symbol": result.symbol,
            "side": result.side,
            "price": result.filled_price,
            "quantity": result.filled_qty,
            "notional": result.notional,
            "pnl": 0.0,       # PnL computed on close by risk_manager
            "strategy": order.strategy,
            "order_id": result.order_id,
            "paper": int(result.paper),
        }
        self.storage.log_trade(trade_record)

    @staticmethod
    def _error_result(order: Order, error: str) -> OrderResult:
        return OrderResult(
            success=False,
            order_id="",
            symbol=order.symbol,
            side=order.side,
            filled_price=0.0,
            filled_qty=0.0,
            commission=0.0,
            timestamp=utc_now(),
            paper=False,
            error=error,
        )
