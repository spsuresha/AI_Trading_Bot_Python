"""
Streamlit monitoring dashboard.
Run with:  streamlit run monitoring/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running via streamlit
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

from config.settings import settings
from data_pipeline.storage import DataStorage
from utils.helpers import safe_divide

# ─────────────────────── page config ─────────────────────────

st.set_page_config(
    page_title="AI Trading Bot – Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────── helpers ─────────────────────────────

@st.cache_resource
def get_storage() -> DataStorage:
    return DataStorage()


def load_trade_log(storage: DataStorage) -> pd.DataFrame:
    return storage.load_trade_log()


# ─────────────────────── sidebar ─────────────────────────────

def render_sidebar() -> None:
    st.sidebar.title("AI Trading Bot")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Exchange:** " + settings.exchange.exchange_id.upper())
    st.sidebar.markdown("**Mode:** " + ("PAPER" if settings.trading.paper_trading else "LIVE"))
    st.sidebar.markdown("**Timeframe:** " + settings.trading.timeframe)
    st.sidebar.markdown("**Symbols:**")
    for s in settings.trading.symbols:
        st.sidebar.markdown(f"  - {s}")


# ─────────────────────── main ────────────────────────────────

def run_dashboard() -> None:
    render_sidebar()

    st.title("AI Trading Bot Dashboard")

    storage = get_storage()
    df_trades = load_trade_log(storage)

    # ── KPI cards ──────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    initial_cap = settings.risk.initial_capital_inr
    total_pnl = float(df_trades["pnl"].sum()) if not df_trades.empty else 0.0
    current_cap = initial_cap + total_pnl
    total_return_pct = safe_divide(total_pnl, initial_cap) * 100
    total_trades = len(df_trades)
    win_rate = (
        safe_divide((df_trades["pnl"] > 0).sum(), total_trades) * 100
        if total_trades > 0 else 0.0
    )

    col1.metric("Capital", f"₹{current_cap:,.2f}", f"{total_return_pct:+.2f}%")
    col2.metric("Total PnL", f"₹{total_pnl:,.2f}")
    col3.metric("Total Trades", str(total_trades))
    col4.metric("Win Rate", f"{win_rate:.1f}%")

    # Max drawdown from equity curve
    if not df_trades.empty and "pnl" in df_trades.columns:
        cum_pnl = df_trades["pnl"].cumsum()
        equity = initial_cap + cum_pnl
        roll_max = equity.cummax()
        drawdown = ((equity - roll_max) / roll_max * 100).min()
        col5.metric("Max Drawdown", f"{abs(drawdown):.2f}%")
    else:
        col5.metric("Max Drawdown", "–")

    st.markdown("---")

    # ── Equity curve ───────────────────────────────────────────
    st.subheader("Equity Curve")
    if not df_trades.empty and "pnl" in df_trades.columns:
        df_trades_sorted = df_trades.sort_values("timestamp")
        equity_series = initial_cap + df_trades_sorted["pnl"].cumsum()
        equity_series.index = pd.to_datetime(df_trades_sorted["timestamp"].values)
        st.line_chart(equity_series, use_container_width=True, height=300)
    else:
        st.info("No trade history yet. Start the bot to see the equity curve.")

    # ── Drawdown chart ─────────────────────────────────────────
    st.subheader("Drawdown (%)")
    if not df_trades.empty:
        equity_series_dd = initial_cap + df_trades_sorted["pnl"].cumsum()
        roll_max_dd = equity_series_dd.cummax()
        dd_series = (equity_series_dd - roll_max_dd) / roll_max_dd * 100
        dd_series.index = pd.to_datetime(df_trades_sorted["timestamp"].values)
        st.area_chart(dd_series, use_container_width=True, height=200)
    else:
        st.info("No data.")

    # ── Strategy performance ───────────────────────────────────
    st.subheader("Strategy Performance")
    if not df_trades.empty and "strategy" in df_trades.columns:
        strat_df = (
            df_trades.groupby("strategy")
            .agg(
                total_pnl=("pnl", "sum"),
                trades=("pnl", "count"),
                win_rate=("pnl", lambda x: safe_divide((x > 0).sum(), len(x)) * 100),
                avg_pnl=("pnl", "mean"),
            )
            .reset_index()
        )
        strat_df = strat_df.rename(columns={
            "strategy": "Strategy",
            "total_pnl": "Total PnL",
            "trades": "Trades",
            "win_rate": "Win Rate %",
            "avg_pnl": "Avg PnL",
        })
        st.dataframe(strat_df.style.format({
            "Total PnL": "{:.2f}",
            "Win Rate %": "{:.1f}",
            "Avg PnL": "{:.4f}",
        }), use_container_width=True)
    else:
        st.info("No strategy data available.")

    # ── Symbol performance ─────────────────────────────────────
    st.subheader("Symbol Performance")
    if not df_trades.empty and "symbol" in df_trades.columns:
        sym_df = (
            df_trades.groupby("symbol")
            .agg(
                total_pnl=("pnl", "sum"),
                trades=("pnl", "count"),
                win_rate=("pnl", lambda x: safe_divide((x > 0).sum(), len(x)) * 100),
            )
            .reset_index()
            .sort_values("total_pnl", ascending=False)
        )
        st.dataframe(sym_df, use_container_width=True)
    else:
        st.info("No symbol data available.")

    # ── PnL distribution ───────────────────────────────────────
    st.subheader("PnL Distribution")
    if not df_trades.empty:
        import plotly.express as px
        fig = px.histogram(
            df_trades, x="pnl", nbins=40,
            color_discrete_sequence=["#00b4d8"],
            title="Trade PnL Distribution",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No PnL data available.")

    # ── Trade history table ────────────────────────────────────
    st.subheader("Trade History")
    if not df_trades.empty:
        display_cols = [c for c in ["timestamp", "symbol", "side", "price", "quantity", "pnl", "strategy"] if c in df_trades.columns]
        st.dataframe(
            df_trades[display_cols].sort_values("timestamp", ascending=False).head(100),
            use_container_width=True,
        )
    else:
        st.info("No trades logged yet.")

    # ── Auto-refresh ───────────────────────────────────────────
    st.markdown("---")
    if st.button("Refresh"):
        st.cache_resource.clear()
        st.rerun()


# Entry point when run as a script
if __name__ == "__main__":
    run_dashboard()
