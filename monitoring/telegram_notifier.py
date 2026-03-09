"""
Telegram alert notifications for the AI Trading Bot.

Sends real-time alerts to a Telegram chat via the Bot API.

Setup
─────
1. Create a bot with @BotFather → copy the token.
2. Start a chat with your bot (or add it to a group).
3. Run: python -c "from monitoring.telegram_notifier import TelegramNotifier; TelegramNotifier().get_chat_id()"
   to find your chat_id, then set it in .env.
4. Set in .env:
       TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
       TELEGRAM_CHAT_ID=-1001234567890

Alert events
────────────
• Bot start / graceful shutdown
• Trade opened  (symbol, side, price, qty, SL, TP)
• Trade closed  (SL / TP triggered, PnL)
• Order rejected by risk manager
• Unhandled runtime error
• Daily portfolio summary
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

# Telegram Bot API base URL
_API_BASE = "https://api.telegram.org/bot{token}/{method}"

# Emoji shortcuts
_UP    = "\U0001F7E2"   # 🟢
_DOWN  = "\U0001F534"   # 🔴
_WARN  = "\u26A0\uFE0F" # ⚠️
_INFO  = "\u2139\uFE0F" # ℹ️
_MONEY = "\U0001F4B0"   # 💰
_CHART = "\U0001F4C8"   # 📈
_STOP  = "\U0001F6D1"   # 🛑
_ROBOT = "\U0001F916"   # 🤖
_CLOCK = "\U0001F551"   # 🕑


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class TelegramNotifier:
    """
    Sends formatted HTML messages to a Telegram bot chat.

    All methods are fire-and-forget: failures are logged but never raised,
    so a Telegram outage never interrupts the trading loop.

    Parameters
    ----------
    bot_token : Telegram bot token (defaults to env TELEGRAM_BOT_TOKEN)
    chat_id   : Target chat / channel id (defaults to env TELEGRAM_CHAT_ID)
    enabled   : Master on/off switch (defaults to env TELEGRAM_ENABLED or True
                when both token and chat_id are present)
    timeout   : HTTP request timeout in seconds (default 5)
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id:   Optional[str] = None,
        enabled:   Optional[bool] = None,
        timeout:   int = 5,
    ) -> None:
        self._token   = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id   or os.getenv("TELEGRAM_CHAT_ID",   "")
        self._timeout = timeout

        if enabled is not None:
            self._enabled = enabled
        else:
            env_val = os.getenv("TELEGRAM_ENABLED", "").lower()
            if env_val in ("0", "false", "no"):
                self._enabled = False
            else:
                self._enabled = bool(self._token and self._chat_id)

        if self._enabled:
            logger.info("TelegramNotifier enabled (chat_id=%s)", self._chat_id)
        else:
            logger.info(
                "TelegramNotifier disabled – set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in .env to enable"
            )

    # ------------------------------------------------------------------ #
    #  Public properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def is_enabled(self) -> bool:
        return self._enabled and bool(self._token) and bool(self._chat_id)

    # ------------------------------------------------------------------ #
    #  Core send                                                           #
    # ------------------------------------------------------------------ #

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a raw message to the configured chat.

        Returns True on success, False on any failure.
        """
        if not self.is_enabled:
            return False

        url = _API_BASE.format(token=self._token, method="sendMessage")
        payload = {
            "chat_id":    self._chat_id,
            "text":       message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            if not resp.ok:
                logger.warning(
                    "Telegram send failed: %s – %s", resp.status_code, resp.text[:200]
                )
                return False
            return True
        except requests.exceptions.RequestException as exc:
            logger.warning("Telegram request error: %s", exc)
            return False

    def get_chat_id(self) -> None:
        """
        Print recent update chat IDs – helpful during first-time setup.
        Run once from CLI to discover your chat_id.
        """
        if not self._token:
            print("Set TELEGRAM_BOT_TOKEN in your .env first.")
            return
        url = _API_BASE.format(token=self._token, method="getUpdates")
        try:
            resp = requests.get(url, timeout=10)
            updates = resp.json().get("result", [])
            if not updates:
                print("No updates found.  Send a message to your bot first, then re-run.")
                return
            for upd in updates[-5:]:
                msg = upd.get("message") or upd.get("channel_post", {})
                chat = msg.get("chat", {})
                print(f"chat_id={chat.get('id')}  type={chat.get('type')}  "
                      f"title/name={chat.get('title') or chat.get('first_name')}")
        except Exception as exc:
            print(f"Error: {exc}")

    # ------------------------------------------------------------------ #
    #  Structured alert methods                                            #
    # ------------------------------------------------------------------ #

    def alert_bot_start(
        self,
        mode:    str,
        symbols: List[str],
        timeframe: str,
    ) -> None:
        """Send startup notification."""
        sym_list = " | ".join(symbols)
        msg = (
            f"{_ROBOT} <b>AI Trading Bot started</b>\n"
            f"<b>Mode:</b> <code>{mode.upper()}</code>\n"
            f"<b>Symbols:</b> <code>{sym_list}</code>\n"
            f"<b>Timeframe:</b> <code>{timeframe}</code>\n"
            f"<b>Time:</b> {_now()}"
        )
        self.send(msg)

    def alert_bot_stop(self, summary: Dict[str, Any]) -> None:
        """Send shutdown notification with final portfolio snapshot."""
        equity  = summary.get("total_equity", 0)
        pnl     = summary.get("realised_pnl", 0)
        trades  = summary.get("total_trades", 0)
        win_pct = summary.get("win_rate_pct", 0)
        icon    = _UP if pnl >= 0 else _DOWN
        msg = (
            f"{_STOP} <b>AI Trading Bot stopped</b>\n"
            f"{icon} <b>Realised PnL:</b> <code>{pnl:+.2f}</code>\n"
            f"<b>Equity:</b> <code>{equity:.2f}</code>\n"
            f"<b>Trades:</b> <code>{trades}</code>   "
            f"<b>Win rate:</b> <code>{win_pct:.1f}%</code>\n"
            f"<b>Time:</b> {_now()}"
        )
        self.send(msg)

    def alert_trade_opened(
        self,
        symbol:    str,
        side:      str,   # "buy" | "sell"
        price:     float,
        quantity:  float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        strategy:  str = "",
    ) -> None:
        """Alert when a new position is opened."""
        icon = _UP if side.lower() == "buy" else _DOWN
        sl_str = f"{stop_loss:.4f}" if stop_loss else "—"
        tp_str = f"{take_profit:.4f}" if take_profit else "—"
        msg = (
            f"{icon} <b>TRADE OPENED — {symbol}</b>\n"
            f"<b>Side:</b>     <code>{side.upper()}</code>\n"
            f"<b>Price:</b>    <code>{price:.4f}</code>\n"
            f"<b>Qty:</b>      <code>{quantity:.6f}</code>\n"
            f"<b>SL:</b>       <code>{sl_str}</code>\n"
            f"<b>TP:</b>       <code>{tp_str}</code>\n"
        )
        if strategy:
            msg += f"<b>Strategy:</b> <code>{strategy}</code>\n"
        msg += f"<b>Time:</b> {_now()}"
        self.send(msg)

    def alert_trade_closed(
        self,
        symbol:  str,
        trigger: str,   # "stop_loss" | "take_profit" | "manual"
        price:   float,
        pnl:     float,
        pnl_pct: Optional[float] = None,
    ) -> None:
        """Alert when a position is closed."""
        icon = _UP if pnl >= 0 else _DOWN
        trigger_label = {
            "stop_loss":   f"{_WARN} Stop-Loss",
            "take_profit": f"{_MONEY} Take-Profit",
        }.get(trigger, trigger.replace("_", " ").title())
        pnl_pct_str = f"  ({pnl_pct:+.2f}%)" if pnl_pct is not None else ""
        msg = (
            f"{icon} <b>TRADE CLOSED — {symbol}</b>\n"
            f"<b>Trigger:</b> {trigger_label}\n"
            f"<b>Price:</b>   <code>{price:.4f}</code>\n"
            f"<b>PnL:</b>     <code>{pnl:+.4f}{pnl_pct_str}</code>\n"
            f"<b>Time:</b>    {_now()}"
        )
        self.send(msg)

    def alert_order_rejected(self, symbol: str, reason: str) -> None:
        """Alert when the risk manager rejects an order."""
        msg = (
            f"{_WARN} <b>ORDER REJECTED — {symbol}</b>\n"
            f"<b>Reason:</b> <code>{reason}</code>\n"
            f"<b>Time:</b>   {_now()}"
        )
        self.send(msg)

    def alert_signal(
        self,
        symbol:    str,
        direction: str,   # "BUY" | "SELL"
        score:     float,
        strategies: Dict[str, str],
    ) -> None:
        """Alert when a strong combined signal is produced (before risk check)."""
        icon = _UP if direction == "BUY" else _DOWN
        strat_lines = "\n".join(
            f"  • {name}: <code>{sig}</code>" for name, sig in strategies.items()
        )
        msg = (
            f"{icon} <b>SIGNAL — {symbol}: {direction}</b>\n"
            f"<b>Score:</b> <code>{score:.3f}</code>\n"
            f"<b>Strategies:</b>\n{strat_lines}\n"
            f"<b>Time:</b> {_now()}"
        )
        self.send(msg)

    def alert_error(self, error: str, context: str = "") -> None:
        """Alert on unhandled runtime error."""
        # Trim very long tracebacks
        short = textwrap.shorten(error, width=400, placeholder=" …")
        msg = (
            f"{_WARN} <b>BOT ERROR</b>\n"
        )
        if context:
            msg += f"<b>Context:</b> <code>{context}</code>\n"
        msg += (
            f"<b>Error:</b>\n<pre>{short}</pre>\n"
            f"<b>Time:</b> {_now()}"
        )
        self.send(msg)

    def alert_daily_summary(self, summary: Dict[str, Any]) -> None:
        """Send a scheduled daily portfolio summary."""
        equity   = summary.get("total_equity",  0)
        pnl      = summary.get("realised_pnl",  0)
        open_pos = summary.get("open_positions", 0)
        trades   = summary.get("total_trades",  0)
        win_pct  = summary.get("win_rate_pct",  0)
        unreal   = summary.get("unrealised_pnl", 0)
        icon     = _UP if pnl >= 0 else _DOWN
        msg = (
            f"{_CHART} <b>Daily Portfolio Summary</b>\n"
            f"────────────────────\n"
            f"{icon} <b>Realised PnL:</b>   <code>{pnl:+.2f}</code>\n"
            f"<b>Unrealised PnL:</b> <code>{unreal:+.2f}</code>\n"
            f"<b>Total equity:</b>  <code>{equity:.2f}</code>\n"
            f"<b>Open positions:</b><code>{open_pos}</code>\n"
            f"<b>Total trades:</b>  <code>{trades}</code>\n"
            f"<b>Win rate:</b>      <code>{win_pct:.1f}%</code>\n"
            f"<b>Time:</b> {_now()}"
        )
        self.send(msg)


# ─────────────────────── singleton ───────────────────────────

_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Return the module-level singleton notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
