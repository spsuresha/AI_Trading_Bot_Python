"""
CLI entry point for GA strategy optimisation.

Workflow
--------
1. Load OHLCV data for all configured symbols from SQLite.
2. Split each symbol's data into TRAIN (70%) / VALIDATION (30%) sets.
3. Run the GA on TRAIN data.
4. Evaluate the best chromosome on the held-out VALIDATION set.
5. Compare against default parameters (baseline).
6. Save the optimised parameters to:
      saved_models/optimized_params.json
7. Print a detailed comparison report to stdout / log.

Usage
-----
    # Standard run (sequential, 50 pop, 30 gen)
    python optimization/optimize.py

    # Faster run for quick testing
    python optimization/optimize.py --pop 20 --gen 10

    # Parallel run with 4 workers
    python optimization/optimize.py --workers 4 --pop 60 --gen 40

    # Target specific symbols
    python optimization/optimize.py --symbols BTC/USDT ETH/USDT

    # Use adaptive mutation variant
    python optimization/optimize.py --adaptive

IMPORTANT (Windows):  This script uses multiprocessing.  Always run it as
    python optimization/optimize.py
NOT with `python -m` or inside a Jupyter notebook without the
`if __name__ == '__main__':` guard (already present at the bottom).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path when the script is run directly
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config.settings import settings
from data_pipeline.storage import DataStorage
from optimization.chromosome import Chromosome
from optimization.fitness import (
    FitnessEvaluator,
    compute_features,
    compute_signals,
    compute_metrics,
    fitness_from_metrics,
    simulate_trades,
    INVALID_FITNESS,
)
from optimization.genetic_optimizer import GAOptimizer, AdaptiveGAOptimizer, OptimisationResult
from utils.logger import get_logger

logger = get_logger(__name__)

RESULTS_PATH = settings.model_dir / "optimized_params.json"


# ─────────────────────── data helpers ────────────────────────


def load_data(
    symbols: Optional[List[str]] = None,
    timeframe: Optional[str]     = None,
    min_candles: int              = 500,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV data from SQLite for the requested symbols."""
    symbols   = symbols   or settings.trading.symbols
    timeframe = timeframe or settings.trading.timeframe
    storage   = DataStorage()
    data: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        df = storage.load_ohlcv(sym, timeframe)
        if df.empty:
            logger.warning("No data for %s — run `python main.py --mode fetch` first", sym)
            continue
        if len(df) < min_candles:
            logger.warning(
                "Only %d candles for %s (need %d) — skipping", len(df), sym, min_candles
            )
            continue
        data[sym] = df
        logger.info("Loaded %d candles for %s", len(df), sym)

    return data


def train_val_split(
    data: Dict[str, pd.DataFrame],
    train_frac: float = 0.70,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Split each symbol's DataFrame into train / validation sets."""
    train_data: Dict[str, pd.DataFrame] = {}
    val_data:   Dict[str, pd.DataFrame] = {}

    for sym, df in data.items():
        n_train = int(len(df) * train_frac)
        train_data[sym] = df.iloc[:n_train].copy()
        val_data[sym]   = df.iloc[n_train:].copy()
        logger.info(
            "%s  train=%d  val=%d", sym, len(train_data[sym]), len(val_data[sym])
        )

    return train_data, val_data


# ─────────────────────── baseline evaluation ─────────────────


def evaluate_default_params(
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    initial_capital: float,
) -> Tuple[float, dict]:
    """Evaluate the current default settings as a baseline."""
    default_chrom = Chromosome.from_default_settings()
    evaluator     = FitnessEvaluator(data, symbols, initial_capital)
    return evaluator.evaluate_with_metrics(default_chrom)


# ─────────────────────── save / load results ─────────────────


def save_results(result: OptimisationResult, symbols: List[str]) -> Path:
    """Persist the optimised parameters and metrics to JSON."""
    output = {
        "optimized_at":       datetime.now(tz=timezone.utc).isoformat(),
        "symbols":            symbols,
        "timeframe":          settings.trading.timeframe,
        "best_fitness":       round(result.best_fitness, 6),
        "train_metrics":      result.train_metrics,
        "validation_metrics": result.validation_metrics,
        "params":             result.best_params,
        "convergence": [
            {
                "generation":   s.generation,
                "best_fitness": s.best_fitness,
                "mean_fitness": s.mean_fitness,
                "std_fitness":  s.std_fitness,
            }
            for s in result.history
        ],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Optimised parameters saved → %s", RESULTS_PATH)
    return RESULTS_PATH


def load_optimized_params() -> Optional[dict]:
    """Load previously saved optimised params (returns None if not found)."""
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text())


# ─────────────────────── report ──────────────────────────────


def print_report(
    result:          OptimisationResult,
    default_train_f: float,
    default_train_m: dict,
    symbols:         List[str],
) -> None:
    """Print a detailed comparison report to stdout."""
    sep  = "─" * 65
    sep2 = "═" * 65

    def _row(label, default_val, opt_val, higher_better=True):
        diff = opt_val - default_val
        arrow = ("↑" if diff > 0 else "↓") if higher_better else \
                ("↓" if diff > 0 else "↑")
        color = ""
        print(f"  {label:<28} {default_val:>9.3f}   {opt_val:>9.3f}   {arrow} {abs(diff):.3f}")

    print("\n" + sep2)
    print("  GA STRATEGY OPTIMISATION REPORT")
    print(sep2)
    print(f"  Symbols  : {', '.join(symbols)}")
    print(f"  Timeframe: {settings.trading.timeframe}")
    print(f"  Best GA fitness (train)    : {result.best_fitness:.4f}")
    print(f"  Default  fitness (train)   : {default_train_f:.4f}")
    improvement = result.best_fitness - default_train_f
    print(f"  Fitness improvement        : {improvement:+.4f}  "
          f"({'better' if improvement > 0 else 'WORSE'})")
    print(sep)

    tm = result.train_metrics
    vm = result.validation_metrics
    dm = default_train_m

    if tm and dm:
        print("\n  METRIC                      DEFAULT     OPTIMISED   DELTA")
        print(sep)
        _row("Sharpe ratio",           dm.get("sharpe_ratio",      0),  tm.get("sharpe_ratio",      0))
        _row("Sortino ratio",          dm.get("sortino_ratio",     0),  tm.get("sortino_ratio",     0))
        _row("Profit factor",          dm.get("profit_factor",     0),  tm.get("profit_factor",     0))
        _row("Win rate (%)",           dm.get("win_rate_pct",      0),  tm.get("win_rate_pct",      0))
        _row("Total return (%)",       dm.get("total_return_pct",  0),  tm.get("total_return_pct",  0))
        _row("CAGR (%)",               dm.get("cagr_pct",          0),  tm.get("cagr_pct",          0))
        _row("Max drawdown (%)",       dm.get("max_drawdown_pct",  0),  tm.get("max_drawdown_pct",  0), higher_better=False)
        _row("Total trades",           dm.get("total_trades",      0),  tm.get("total_trades",      0))

    if vm:
        print(f"\n{sep}")
        print("  VALIDATION SET METRICS (out-of-sample)")
        print(sep)
        for k, v in vm.items():
            if isinstance(v, float):
                print(f"  {k:<28} {v:.4f}")
            else:
                print(f"  {k:<28} {v}")

    print(f"\n{sep}")
    print("  OPTIMISED PARAMETERS")
    print(sep)
    p = result.best_params
    print(f"  RSI period         : {p['rsi_period']}")
    print(f"  MACD               : fast={p['macd_fast']} slow={p['macd_slow']} signal={p['macd_signal']}")
    print(f"  ATR period         : {p['atr_period']}")
    print(f"  Bollinger          : period={p['bb_period']}  std={p['bb_std']:.2f}")
    print(f"  Volume MA period   : {p['volume_ma_period']}")
    print(f"  Momentum lookback  : {p['momentum_lookback']}")
    print(f"  Z-score threshold  : {p['zscore_threshold']:.2f}")
    print(f"  Breakout lookback  : {p['breakout_lookback']}")
    print(f"  Signal threshold   : {p['signal_threshold']:.3f}")
    print(f"  Strategy weights   :")
    for name, w in p["weights"].items():
        print(f"    {name:<22} {w:.4f}")
    print(f"  Stop-loss ATR mult : {p['stop_loss_atr_mult']:.2f}×")
    print(f"  Take-profit ATR mult:{p['take_profit_atr_mult']:.2f}×  "
          f"(R:R = 1:{p['take_profit_atr_mult']/p['stop_loss_atr_mult']:.1f})")
    print(f"  Risk per trade     : {p['risk_per_trade_pct']*100:.2f}%")

    print(f"\n{sep}")
    print("  CONVERGENCE HISTORY")
    print(sep)
    print(f"  {'Gen':>4}  {'Best':>8}  {'Mean':>8}  {'Std':>7}")
    for s in result.history:
        print(f"  {s.generation:>4}  {s.best_fitness:>8.4f}  "
              f"{s.mean_fitness:>8.4f}  {s.std_fitness:>7.4f}")

    print(sep2)
    print(f"  Results saved to: {RESULTS_PATH}")
    print(sep2 + "\n")


# ─────────────────────── main ────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise strategy parameters using a genetic algorithm"
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=None,
        help="Symbols to optimise (default: all configured symbols)",
    )
    parser.add_argument(
        "--timeframe", default=None,
        help="OHLCV timeframe (default: from settings)",
    )
    parser.add_argument(
        "--pop", type=int, default=50,
        help="GA population size (default: 50)",
    )
    parser.add_argument(
        "--gen", type=int, default=30,
        help="Maximum generations (default: 30)",
    )
    parser.add_argument(
        "--elite", type=int, default=5,
        help="Elite individuals preserved each generation (default: 5)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes (default: 1 = sequential)",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early-stop patience in generations (default: 10)",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.05,
        help="Mutation sigma in [0,1] space (default: 0.05)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.70,
        help="Fraction of data used for training (default: 0.70)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Use adaptive mutation (boosts sigma when diversity is low)",
    )
    parser.add_argument(
        "--capital", type=float, default=settings.risk.initial_capital_inr,
        help="Initial capital for simulation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    symbols = args.symbols or settings.trading.symbols
    logger.info("Target symbols: %s", symbols)

    # ── Load data ─────────────────────────────────────────────────
    data = load_data(symbols=symbols, timeframe=args.timeframe)
    if not data:
        logger.error(
            "No data loaded.  Run `python main.py --mode fetch` to download data first."
        )
        sys.exit(1)

    active_symbols = list(data.keys())

    # ── Train / validation split ──────────────────────────────────
    train_data, val_data = train_val_split(data, train_frac=args.train_frac)

    # ── Baseline: evaluate default params ────────────────────────
    logger.info("Evaluating default parameter baseline …")
    default_train_f, default_train_m = evaluate_default_params(
        train_data, active_symbols, args.capital
    )
    logger.info("Default fitness (train): %.4f", default_train_f)

    # ── Build GA optimiser ────────────────────────────────────────
    OptimizerClass = AdaptiveGAOptimizer if args.adaptive else GAOptimizer
    optimizer = OptimizerClass(
        pop_size       = args.pop,
        n_generations  = args.gen,
        n_elite        = args.elite,
        patience       = args.patience,
        n_workers      = args.workers,
        mutation_sigma = args.sigma,
        seed           = args.seed,
    )

    # ── Run GA on training data ───────────────────────────────────
    result = optimizer.run(
        df_dict         = train_data,
        symbols         = active_symbols,
        initial_capital = args.capital,
        commission_pct  = settings.backtest.commission_pct,
        slippage_pct    = settings.backtest.slippage_pct,
        progress_cb     = _log_progress,
    )

    # ── Validate best chromosome on held-out data ─────────────────
    logger.info("Evaluating best chromosome on validation set …")
    val_evaluator = FitnessEvaluator(
        val_data, active_symbols, args.capital
    )
    _, val_metrics = val_evaluator.evaluate_with_metrics(result.best_chromosome)
    result.validation_metrics = val_metrics

    val_fitness = fitness_from_metrics(val_metrics)
    logger.info("Validation fitness: %.4f", val_fitness)

    # ── Save results ──────────────────────────────────────────────
    save_results(result, active_symbols)

    # ── Print report ──────────────────────────────────────────────
    print_report(result, default_train_f, default_train_m, active_symbols)


def _log_progress(stats) -> None:
    """Progress callback – printed inline by the GA loop."""
    pass   # GAOptimizer already logs each generation; this is a hook for UIs


# ─────────────────────── Windows guard ───────────────────────
# REQUIRED: multiprocessing.Pool uses 'spawn' on Windows, which re-imports
# this module in each worker.  Without this guard the main() function would
# be executed recursively in every worker process.

if __name__ == "__main__":
    main()
