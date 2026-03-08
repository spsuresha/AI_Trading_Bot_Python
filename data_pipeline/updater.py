"""
Incremental data updater.
Fetches only the candles newer than what is already stored.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from config.settings import settings
from data_pipeline.collector import MarketDataCollector
from data_pipeline.storage import DataStorage
from utils.logger import get_logger

logger = get_logger(__name__)


class DataUpdater:
    """Keeps local SQLite up-to-date with new candles from the exchange."""

    def __init__(
        self,
        collector: Optional[MarketDataCollector] = None,
        storage: Optional[DataStorage] = None,
    ) -> None:
        self.collector = collector or MarketDataCollector()
        self.storage = storage or DataStorage()

    def update_symbol(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch only new candles for a symbol and append them to storage.

        Returns the freshly fetched DataFrame (may be empty if no new data).
        """
        timeframe = timeframe or settings.trading.timeframe
        last_ts = self.storage.get_latest_timestamp(symbol, timeframe)

        since_ms: Optional[int] = None
        if last_ts:
            # Add one candle-width to avoid re-fetching the last stored candle
            interval_ms = _timeframe_to_ms(timeframe)
            since_ms = int(pd.Timestamp(last_ts).timestamp() * 1000) + interval_ms
            logger.debug("Incremental update for %s from %s", symbol, last_ts)
        else:
            logger.info("No existing data for %s – full initial fetch", symbol)

        df_new = self.collector.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=settings.trading.lookback_candles,
            since=since_ms,
        )

        if df_new.empty:
            logger.info("No new candles for %s", symbol)
            return df_new

        self.storage.save_ohlcv(symbol, timeframe, df_new)
        logger.info("Updated %d new candles for %s", len(df_new), symbol)
        return df_new

    def update_all(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Update all configured symbols."""
        symbols = symbols or settings.trading.symbols
        results: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.update_symbol(sym, timeframe=timeframe)
            results[sym] = df
        return results

    def get_latest_data(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        candles: int = 500,
    ) -> pd.DataFrame:
        """
        Return the most recent `candles` rows for a symbol.
        Runs an incremental update first to ensure data is fresh.
        """
        timeframe = timeframe or settings.trading.timeframe
        self.update_symbol(symbol, timeframe=timeframe)
        df = self.storage.load_ohlcv(symbol, timeframe)
        return df.tail(candles) if not df.empty else df


# ─────────────────────── helpers ─────────────────────────────


def _timeframe_to_ms(timeframe: str) -> int:
    """Convert ccxt timeframe string to milliseconds."""
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    unit = timeframe[-1]
    num = int(timeframe[:-1])
    return num * units.get(unit, 3_600_000)
