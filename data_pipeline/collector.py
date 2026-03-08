"""
Market data collector using ccxt.
Supports Binance, CoinDCX (via Binance-compatible interface), and any ccxt exchange.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import ccxt
import pandas as pd

from config.settings import settings
from utils.logger import get_logger
from utils.helpers import validate_ohlcv

logger = get_logger(__name__)


class MarketDataCollector:
    """Fetches OHLCV data from a ccxt-supported exchange."""

    OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self) -> None:
        cfg = settings.exchange
        exchange_class = getattr(ccxt, cfg.exchange_id)

        params: dict = {
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "enableRateLimit": cfg.rate_limit,
        }
        if cfg.testnet:
            params["options"] = {"defaultType": "future"}

        self.exchange: ccxt.Exchange = exchange_class(params)

        if cfg.testnet and hasattr(self.exchange, "set_sandbox_mode"):
            try:
                self.exchange.set_sandbox_mode(True)
                logger.info("Sandbox / testnet mode enabled")
            except Exception:
                pass

        logger.info("Initialised exchange: %s", cfg.exchange_id)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a single symbol.

        Parameters
        ----------
        symbol    : e.g. "BTC/USDT"
        timeframe : ccxt timeframe string ("1m", "5m", "1h", "1d", …)
        limit     : number of candles
        since     : start timestamp in milliseconds (optional)

        Returns
        -------
        pd.DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
        """
        logger.debug("Fetching %s candles for %s @ %s", limit, symbol, timeframe)
        try:
            raw: List[List] = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit, since=since
            )
        except ccxt.NetworkError as exc:
            logger.error("Network error fetching %s: %s", symbol, exc)
            return pd.DataFrame()
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error fetching %s: %s", symbol, exc)
            return pd.DataFrame()

        if not raw:
            logger.warning("Empty OHLCV response for %s", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=self.OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        df.sort_index(inplace=True)
        df = validate_ohlcv(df)
        logger.info("Fetched %d candles for %s", len(df), symbol)
        return df

    def fetch_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for all configured symbols."""
        symbols = symbols or settings.trading.symbols
        timeframe = timeframe or settings.trading.timeframe
        limit = limit or settings.trading.lookback_candles

        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            if not df.empty:
                result[sym] = df
            time.sleep(self.exchange.rateLimit / 1000)  # honour rate limit

        logger.info("Fetched data for %d / %d symbols", len(result), len(symbols))
        return result

    def fetch_ticker(self, symbol: str) -> dict:
        """Return latest ticker data for a symbol."""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as exc:
            logger.error("Error fetching ticker for %s: %s", symbol, exc)
            return {}

    def fetch_order_book(self, symbol: str, depth: int = 10) -> dict:
        """Return order book snapshot."""
        try:
            return self.exchange.fetch_order_book(symbol, limit=depth)
        except Exception as exc:
            logger.error("Error fetching order book for %s: %s", symbol, exc)
            return {}

    def get_exchange_info(self, symbol: str) -> dict:
        """Return market info (lot size, tick size, etc.)."""
        try:
            markets = self.exchange.load_markets()
            return markets.get(symbol, {})
        except Exception as exc:
            logger.error("Error loading markets: %s", exc)
            return {}
