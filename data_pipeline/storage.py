"""
Persistent storage layer.
Supports both SQLite (default) and CSV flat files.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DataStorage:
    """
    Read / write OHLCV data to SQLite.
    Each symbol × timeframe combination gets its own table.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("DataStorage initialised: %s", self.db_path)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_registry (
                    symbol      TEXT NOT NULL,
                    timeframe   TEXT NOT NULL,
                    table_name  TEXT NOT NULL,
                    last_update TEXT,
                    PRIMARY KEY (symbol, timeframe)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT,
                    symbol        TEXT,
                    side          TEXT,
                    price         REAL,
                    quantity      REAL,
                    notional      REAL,
                    pnl           REAL,
                    strategy      TEXT,
                    order_id      TEXT,
                    paper         INTEGER DEFAULT 1
                )
            """)

    @staticmethod
    def _table_name(symbol: str, timeframe: str) -> str:
        safe = symbol.replace("/", "_").replace("-", "_").lower()
        return f"ohlcv_{safe}_{timeframe}"

    # ------------------------------------------------------------------ #
    #  OHLCV                                                               #
    # ------------------------------------------------------------------ #

    def save_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Upsert OHLCV rows (insert-or-replace by timestamp index)."""
        if df.empty:
            return

        table = self._table_name(symbol, timeframe)
        df_save = df.copy()
        df_save.index = df_save.index.astype(str)
        df_save.index.name = "timestamp"

        with self._connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS "{table}" (
                    timestamp TEXT PRIMARY KEY,
                    open      REAL,
                    high      REAL,
                    low       REAL,
                    close     REAL,
                    volume    REAL
                )
            """)
            df_save.to_sql(table, conn, if_exists="replace", index=True)

            conn.execute("""
                INSERT INTO data_registry (symbol, timeframe, table_name, last_update)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(symbol, timeframe) DO UPDATE SET
                    last_update = excluded.last_update
            """, (symbol, timeframe, table))

        logger.debug("Saved %d rows → %s", len(df), table)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from SQLite into a DataFrame."""
        table = self._table_name(symbol, timeframe)
        with self._connect() as conn:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )["name"].tolist()
            if table not in tables:
                logger.warning("Table %s not found in DB", table)
                return pd.DataFrame()

            query = f'SELECT * FROM "{table}"'
            conditions = []
            if start:
                conditions.append(f"timestamp >= '{start}'")
            if end:
                conditions.append(f"timestamp <= '{end}'")
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp"

            df = pd.read_sql(query, conn, index_col="timestamp", parse_dates=["timestamp"])

        df.index = pd.to_datetime(df.index, utc=True)
        logger.debug("Loaded %d rows from %s", len(df), table)
        return df

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[str]:
        """Return the most recent timestamp stored for a symbol / timeframe."""
        table = self._table_name(symbol, timeframe)
        with self._connect() as conn:
            try:
                row = conn.execute(
                    f'SELECT MAX(timestamp) FROM "{table}"'
                ).fetchone()
                return row[0] if row else None
            except sqlite3.OperationalError:
                return None

    # ------------------------------------------------------------------ #
    #  CSV helpers                                                         #
    # ------------------------------------------------------------------ #

    def save_csv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
        safe = symbol.replace("/", "_")
        path = settings.data_dir / f"{safe}_{timeframe}.csv"
        df.to_csv(path)
        logger.info("Saved CSV: %s", path)
        return path

    def load_csv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        safe = symbol.replace("/", "_")
        path = settings.data_dir / f"{safe}_{timeframe}.csv"
        if not path.exists():
            logger.warning("CSV not found: %s", path)
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    # ------------------------------------------------------------------ #
    #  Trade log                                                           #
    # ------------------------------------------------------------------ #

    def log_trade(self, trade: dict) -> None:
        """Persist a single trade record."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO trade_log
                    (timestamp, symbol, side, price, quantity, notional, pnl, strategy, order_id, paper)
                VALUES
                    (:timestamp, :symbol, :side, :price, :quantity, :notional, :pnl, :strategy, :order_id, :paper)
            """, trade)

    def load_trade_log(self, symbol: Optional[str] = None) -> pd.DataFrame:
        with self._connect() as conn:
            if symbol:
                df = pd.read_sql(
                    "SELECT * FROM trade_log WHERE symbol = ? ORDER BY timestamp",
                    conn,
                    params=(symbol,),
                )
            else:
                df = pd.read_sql(
                    "SELECT * FROM trade_log ORDER BY timestamp", conn
                )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
