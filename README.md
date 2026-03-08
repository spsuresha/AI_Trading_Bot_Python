# AI Trading Bot – Python

A production-grade, modular AI-powered cryptocurrency trading system designed
for a ₹1 lakh (~$1 200 USD) account. Supports paper trading and live execution
via any ccxt-compatible exchange (Binance, CoinDCX, etc.).

---

## Project Structure

```
AI_trading_bot_python/
├── config/               # Central configuration (settings.py)
├── data_pipeline/        # Market data collection, storage, incremental updates
│   ├── collector.py      # ccxt OHLCV fetcher
│   ├── storage.py        # SQLite + CSV persistence
│   └── updater.py        # Incremental data update logic
├── features/             # Technical indicator computation
│   └── engineer.py       # RSI, EMA, MACD, ATR, Bollinger, Volume, Patterns
├── models/               # Machine learning
│   ├── trainer.py        # XGBoost / RandomForest training pipeline
│   └── predictor.py      # Inference wrapper
├── strategies/           # Trading signal generation
│   ├── base.py           # Abstract base + Signal dataclass
│   ├── momentum.py       # EMA trend + RSI momentum strategy
│   ├── mean_reversion.py # Bollinger Band + Z-score mean reversion
│   ├── breakout.py       # N-bar high/low breakout with volume confirmation
│   ├── ai_prediction.py  # ML model signal wrapper
│   └── engine.py         # Weighted signal aggregation engine
├── risk_management/      # Position sizing, SL/TP, limits
│   └── risk_manager.py
├── backtesting/          # Vectorised historical backtesting
│   └── backtester.py
├── execution/            # Order placement (paper + live)
│   └── executor.py
├── portfolio/            # Real-time position & PnL tracking
│   └── tracker.py
├── monitoring/           # Streamlit dashboard
│   └── dashboard.py
├── utils/                # Logger, helper functions
├── main.py               # Bot entry point (live / backtest / train / fetch)
├── retrain.py            # Weekly model retraining pipeline
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the bot

Edit [config/settings.py](config/settings.py) or set environment variables:

```bash
export EXCHANGE_API_KEY="your_api_key"
export EXCHANGE_API_SECRET="your_api_secret"
```

Key settings to review:
- `ExchangeConfig.exchange_id` – `"binance"` or `"coindcx"`
- `ExchangeConfig.testnet` – `True` for paper trading
- `TradingConfig.symbols` – which pairs to trade
- `TradingConfig.paper_trading` – `True` for simulation

### 3. Fetch historical data

```bash
python main.py --mode fetch
```

### 4. Train the ML model

```bash
python main.py --mode train
```

### 5. Backtest

```bash
python main.py --mode backtest
```

### 6. Run the bot (paper trading by default)

```bash
python main.py --mode live
```

### 7. View the dashboard

```bash
streamlit run monitoring/dashboard.py
```

### 8. Schedule weekly retraining

```bash
python retrain.py --schedule --interval 7
```

---

## Risk Management

| Rule                  | Default Value        |
|-----------------------|----------------------|
| Risk per trade        | 1% of capital        |
| Stop-loss             | 2× ATR from entry    |
| Take-profit           | 4× ATR from entry    |
| Max daily loss        | 3% of capital        |
| Max portfolio drawdown| 10%                  |
| Max open positions    | 4                    |

---

## Strategies

| Strategy        | Logic                                            |
|-----------------|--------------------------------------------------|
| Momentum        | EMA alignment + RSI 50–70 + MACD histogram       |
| Mean Reversion  | Bollinger Bands + Z-score + RSI extremes         |
| Breakout        | N-bar high/low break + volume surge              |
| AI Prediction   | XGBoost / RandomForest direction classifier      |

Signals are combined using configurable weights (default 25% each) and a
threshold filter before any order is placed.

---

## Performance Metrics (backtesting)

- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- CAGR
- Avg Win / Avg Loss

---

## Disclaimer

This software is for educational and research purposes only.
Cryptocurrency trading involves substantial risk of loss.
Never trade with money you cannot afford to lose.
Past performance does not guarantee future results.
