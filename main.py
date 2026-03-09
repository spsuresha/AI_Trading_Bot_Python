"""
Main bot engine.
Runs the primary trading loop: fetch data → generate signals →
apply risk management → execute orders → log results.

Usage:
    python main.py                     # runs the live/paper trading loop
    python main.py --mode backtest     # runs a backtest on stored data
    python main.py --mode train        # trains the ML model
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from config.settings import settings
from data_pipeline.collector import MarketDataCollector
from data_pipeline.updater import DataUpdater
from data_pipeline.storage import DataStorage
from execution.executor import Order, OrderExecutor
from portfolio.tracker import PortfolioTracker
from risk_management.risk_manager import RiskManager, TradeProposal
from strategies.engine import StrategyEngine, SignalDirection
from monitoring.telegram_notifier import get_notifier
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────── graceful shutdown ────────────────────

_RUNNING = True


def _handle_signal(signum, frame):
    global _RUNNING
    logger.info("Shutdown signal received – stopping bot gracefully …")
    _RUNNING = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ─────────────────────── bot class ────────────────────────────

class TradingBot:
    """
    The main orchestrator.
    On each tick:
    1. Fetch / update OHLCV data for all symbols
    2. Run strategy engine → combined signals
    3. For BUY/SELL signals, consult risk manager
    4. Execute approved orders via the execution engine
    5. Check SL/TP for open positions
    6. Update portfolio tracker
    7. Log everything
    """

    LOOP_INTERVAL_SECONDS: int = 60   # poll every 60 s; real candle boundary detected via timestamp

    def __init__(self) -> None:
        self.updater  = DataUpdater()
        self.storage  = DataStorage()
        self.engine   = StrategyEngine()
        self.risk     = RiskManager()
        self.executor = OrderExecutor(storage=self.storage)
        self.portfolio = PortfolioTracker()
        self.notifier  = get_notifier()
        self._last_candle_ts: Dict[str, str] = {}
        self._last_summary_hour: Optional[int] = None
        logger.info("TradingBot initialised")

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        global _RUNNING
        logger.info("=" * 60)
        logger.info("AI Trading Bot starting [%s mode]",
                    "PAPER" if settings.trading.paper_trading else "LIVE")
        logger.info("Symbols: %s", settings.trading.symbols)
        logger.info("Timeframe: %s", settings.trading.timeframe)
        logger.info("=" * 60)

        mode = "PAPER" if settings.trading.paper_trading else "LIVE"
        if settings.telegram.alert_on_trade:
            self.notifier.alert_bot_start(
                mode=mode,
                symbols=settings.trading.symbols,
                timeframe=settings.trading.timeframe,
            )

        while _RUNNING:
            try:
                self._tick()
                self._maybe_send_daily_summary()
            except Exception as exc:
                logger.error("Unhandled exception in tick: %s", exc, exc_info=True)
                if settings.telegram.alert_on_error:
                    self.notifier.alert_error(str(exc), context="main tick loop")

            if _RUNNING:
                logger.debug("Sleeping %ds until next tick …", self.LOOP_INTERVAL_SECONDS)
                time.sleep(self.LOOP_INTERVAL_SECONDS)

        self._shutdown()

    def _tick(self) -> None:
        logger.debug("Tick at %s", datetime.now(tz=timezone.utc).isoformat())

        # 1. Update data
        data: Dict[str, pd.DataFrame] = {}
        for sym in settings.trading.symbols:
            df = self.updater.get_latest_data(sym)
            if df.empty or len(df) < 50:
                logger.warning("Skipping %s – insufficient data", sym)
                continue

            # Only process on a NEW candle
            latest_ts = str(df.index[-1])
            if latest_ts == self._last_candle_ts.get(sym):
                continue
            self._last_candle_ts[sym] = latest_ts
            data[sym] = df

        if not data:
            logger.debug("No new candles this tick")
            return

        # 2. Generate signals
        signals = self.engine.process_all(data)

        # 3. Check SL/TP on open positions
        self._check_exits(data)

        # 4. Process new signals
        for sym, combined in signals.items():
            if combined.direction == SignalDirection.HOLD:
                continue
            if sym not in data:
                continue

            df = data[sym]
            current_price = float(df["close"].iloc[-1])
            atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else current_price * 0.01

            direction_name = combined.direction.name  # "BUY" | "SELL"

            # Telegram: signal alert (before risk check)
            if settings.telegram.alert_on_signal:
                strat_sigs = {
                    k: v.direction.name
                    for k, v in combined.individual_signals.items()
                }
                self.notifier.alert_signal(
                    symbol=sym,
                    direction=direction_name,
                    score=combined.combined_score,
                    strategies=strat_sigs,
                )

            proposal = TradeProposal(
                symbol=sym,
                direction=int(combined.direction),
                entry_price=current_price,
                atr=atr,
                signal_strength=combined.combined_score,
            )
            decision = self.risk.evaluate(proposal)

            if not decision.approved:
                logger.info("[%s] Order REJECTED: %s", sym, decision.rejection_reason)
                if settings.telegram.alert_on_rejection:
                    self.notifier.alert_order_rejected(sym, decision.rejection_reason)
                continue

            # 5. Execute
            side = "buy" if decision.direction == 1 else "sell"
            strat_label = str({k: v.direction.name for k, v in combined.individual_signals.items()})
            order = Order(
                symbol=sym,
                side=side,
                quantity=decision.quantity,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                strategy=strat_label,
            )
            result = self.executor.execute(order, current_price=current_price)

            if result.success:
                self.risk.record_trade_open(decision)
                self.portfolio.open_position(
                    symbol=sym,
                    direction=decision.direction,
                    entry_price=result.filled_price,
                    quantity=result.filled_qty,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                )
                logger.info("Trade opened: %s %s @ %.4f", side.upper(), sym, result.filled_price)

                # Telegram: trade opened
                if settings.telegram.alert_on_trade:
                    self.notifier.alert_trade_opened(
                        symbol=sym,
                        side=side,
                        price=result.filled_price,
                        quantity=result.filled_qty,
                        stop_loss=decision.stop_loss,
                        take_profit=decision.take_profit,
                        strategy=strat_label,
                    )

        # 6. Log portfolio summary
        summary = self.portfolio.get_summary()
        logger.info("Portfolio: equity=%.2f | open=%d | pnl=%.2f | wins=%.1f%%",
                    summary["total_equity"], summary["open_positions"],
                    summary["realised_pnl"], summary["win_rate_pct"])

    def _check_exits(self, data: Dict[str, pd.DataFrame]) -> None:
        """Check SL/TP for all open positions."""
        for sym in list(self.portfolio.positions.keys()):
            if sym not in data:
                continue
            df = data[sym]
            current_price = float(df["close"].iloc[-1])
            self.portfolio.update_prices({sym: current_price})

            trigger = self.risk.check_stop_loss_take_profit(sym, current_price)
            if trigger:
                exit_price = (
                    self.portfolio.positions[sym].stop_loss
                    if trigger == "stop_loss"
                    else self.portfolio.positions[sym].take_profit
                )
                self.risk.record_trade_close(sym, exit_price)
                trade = self.portfolio.close_position(sym, exit_price, reason=trigger)
                if trade:
                    side = "buy" if trade["direction"] == -1 else "sell"
                    order = Order(
                        symbol=sym,
                        side=side,
                        quantity=trade["quantity"],
                        strategy=f"exit_{trigger}",
                    )
                    self.executor.execute(order, current_price=exit_price)
                    logger.info("[%s] %s exit: price=%.4f PnL=%.4f",
                                sym, trigger, exit_price, trade["pnl"])

                    # Telegram: trade closed
                    if settings.telegram.alert_on_trade:
                        entry = trade.get("entry_price") or exit_price
                        pnl_pct = (trade["pnl"] / (entry * trade["quantity"]) * 100) if entry else None
                        self.notifier.alert_trade_closed(
                            symbol=sym,
                            trigger=trigger,
                            price=exit_price,
                            pnl=trade["pnl"],
                            pnl_pct=pnl_pct,
                        )

    def _maybe_send_daily_summary(self) -> None:
        """Send a daily summary once per day at the configured UTC hour."""
        if not settings.telegram.daily_summary:
            return
        now_hour = datetime.now(tz=timezone.utc).hour
        target   = settings.telegram.daily_summary_hour
        if now_hour == target and self._last_summary_hour != target:
            self._last_summary_hour = target
            summary = self.portfolio.get_summary()
            self.notifier.alert_daily_summary(summary)

    def _shutdown(self) -> None:
        logger.info("Shutting down – final portfolio state:")
        summary = self.portfolio.get_summary()
        for k, v in summary.items():
            logger.info("  %-25s: %s", k, v)
        if settings.telegram.alert_on_trade:
            self.notifier.alert_bot_stop(summary)


# ─────────────────────── CLI entry-points ─────────────────────


def run_backtest() -> None:
    """Run a backtest using stored data."""
    from backtesting.backtester import Backtester

    logger.info("Starting backtest …")
    storage = DataStorage()
    backtester = Backtester()

    for sym in settings.trading.symbols:
        df = storage.load_ohlcv(sym, settings.trading.timeframe)
        if df.empty:
            logger.warning("No stored data for %s – fetch data first", sym)
            continue
        try:
            result = backtester.run(df, sym)
            summary = result.summary()
            logger.info("Backtest results for %s:", sym)
            for k, v in summary.items():
                logger.info("  %-25s: %s", k, v)
        except Exception as exc:
            logger.error("Backtest failed for %s: %s", sym, exc)


def run_training() -> None:
    """Train the ML model on stored data."""
    from models.trainer import ModelTrainer

    logger.info("Starting model training …")
    storage = DataStorage()
    data = {}
    for sym in settings.trading.symbols:
        df = storage.load_ohlcv(sym, settings.trading.timeframe)
        if not df.empty:
            data[sym] = df

    if not data:
        logger.error("No data available for training. Run data fetch first.")
        return

    trainer = ModelTrainer()
    metrics = trainer.train(data)
    logger.info("Training completed. Metrics:")
    for k, v in metrics.items():
        if k != "classification_report":
            logger.info("  %-25s: %s", k, v)


def fetch_data() -> None:
    """Fetch and store historical data for all configured symbols."""
    logger.info("Fetching historical data …")
    updater = DataUpdater()
    results = updater.update_all()
    for sym, df in results.items():
        logger.info("  %s: %d new candles", sym, len(df))


# ─────────────────────── argument parsing ─────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "train", "fetch"],
        default="live",
        help="Operation mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode

    if mode == "live":
        bot = TradingBot()
        bot.run()
    elif mode == "backtest":
        run_backtest()
    elif mode == "train":
        run_training()
    elif mode == "fetch":
        fetch_data()
    else:
        logger.error("Unknown mode: %s", mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
