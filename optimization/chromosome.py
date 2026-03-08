"""
Chromosome encoding for strategy parameter optimization.

Each chromosome is a vector of 19 real-valued genes in [0, 1] (normalized
space).  The `decode()` method maps them to actual parameter ranges.

Parameter space covers:
  - Feature engineering  (RSI, MACD, ATR, Bollinger, Volume periods)
  - Strategy parameters  (lookbacks, thresholds, weights)
  - Risk management      (SL/TP multipliers, position-size percentage)

Constraints are baked into the encoding:
  - macd_slow  is encoded as (macd_fast + offset) → always > macd_fast
  - take_profit is encoded as (stop_loss × ratio)  → always > stop_loss
  - strategy weights are decoded then L1-normalized → always sum to 1
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple


# ─────────────────────── gene definition ─────────────────────


class Gene(NamedTuple):
    name: str
    lo: float
    hi: float
    dtype: type       # int  or  float
    description: str


# 19-dimensional parameter space
PARAM_SPACE: List[Gene] = [
    # ── Feature engineering ──────────────────────────────────────
    Gene("rsi_period",         7,    21,    int,   "RSI lookback period"),
    Gene("macd_fast",          6,    16,    int,   "MACD fast EMA period"),
    Gene("macd_slow_offset",   8,    20,    int,   "MACD slow = fast + offset (ensures slow > fast)"),
    Gene("macd_signal",        5,    13,    int,   "MACD signal EMA period"),
    Gene("atr_period",         7,    21,    int,   "ATR lookback period"),
    Gene("bb_period",          10,   30,    int,   "Bollinger Band rolling period"),
    Gene("bb_std",             1.5,  3.0,   float, "Bollinger Band std multiplier"),
    Gene("volume_ma_period",   10,   30,    int,   "Volume moving-average period"),
    # ── Strategy parameters ───────────────────────────────────────
    Gene("momentum_lookback",  5,    40,    int,   "Momentum % return lookback"),
    Gene("zscore_threshold",   1.0,  3.5,   float, "Mean-reversion Z-score entry threshold"),
    Gene("breakout_lookback",  5,    40,    int,   "Breakout N-bar high/low lookback"),
    Gene("signal_threshold",   0.25, 0.75,  float, "Min combined score to generate a trade"),
    # ── Strategy weights (L1-normalised to sum=1 after decode) ────
    Gene("w_momentum",         0.05, 0.60,  float, "Momentum strategy weight (raw)"),
    Gene("w_mean_reversion",   0.05, 0.60,  float, "Mean-reversion strategy weight (raw)"),
    Gene("w_breakout",         0.05, 0.60,  float, "Breakout strategy weight (raw)"),
    Gene("w_ai_prediction",    0.05, 0.60,  float, "AI-prediction strategy weight (raw)"),
    # ── Risk management ───────────────────────────────────────────
    Gene("stop_loss_atr_mult", 1.0,  3.5,   float, "Stop-loss = entry ± N × ATR"),
    Gene("tp_sl_ratio",        1.2,  3.5,   float, "Take-profit = stop_loss × ratio (R:R ≥ 1.2)"),
    Gene("risk_per_trade_pct", 0.005, 0.025, float, "Fraction of capital risked per trade"),
]

N_GENES: int = len(PARAM_SPACE)   # 19


# ─────────────────────── chromosome ──────────────────────────


@dataclass
class Chromosome:
    """
    A single individual in the GA population.

    All `genes` are normalised to [0, 1]; call `decode()` to get
    actual parameter values ready for use in the backtester.
    """

    genes: List[float] = field(default_factory=lambda: [0.5] * N_GENES)

    def __post_init__(self) -> None:
        if len(self.genes) != N_GENES:
            raise ValueError(f"Expected {N_GENES} genes, got {len(self.genes)}")

    # ------------------------------------------------------------------ #
    #  Factories                                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def random(cls, rng: Optional[random.Random] = None) -> "Chromosome":
        """Uniformly random chromosome."""
        _rng = rng or random
        return cls(genes=[_rng.random() for _ in range(N_GENES)])

    @classmethod
    def from_default_settings(cls) -> "Chromosome":
        """
        Initialise a chromosome that encodes the current default settings.
        Useful to seed the initial population with a known-good baseline.
        """
        from config.settings import settings as cfg

        def _norm(val: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 0.5
            return max(0.0, min(1.0, (val - lo) / (hi - lo)))

        total_w = sum(cfg.strategy.weights.values()) or 1.0
        macd_slow_offset = cfg.features.macd_slow - cfg.features.macd_fast
        sl = cfg.risk.stop_loss_atr_mult
        tp_sl = cfg.risk.take_profit_atr_mult / max(sl, 1e-6)

        genes = [
            _norm(cfg.features.rsi_period,            7,    21),
            _norm(cfg.features.macd_fast,              6,    16),
            _norm(macd_slow_offset,                    8,    20),
            _norm(cfg.features.macd_signal,            5,    13),
            _norm(cfg.features.atr_period,             7,    21),
            _norm(cfg.features.bb_period,              10,   30),
            _norm(cfg.features.bb_std,                 1.5,  3.0),
            _norm(cfg.features.volume_ma_period,       10,   30),
            _norm(cfg.strategy.momentum_lookback,      5,    40),
            _norm(cfg.strategy.mean_rev_zscore_threshold, 1.0, 3.5),
            _norm(cfg.strategy.breakout_lookback,      5,    40),
            _norm(cfg.strategy.signal_threshold,       0.25, 0.75),
            _norm(cfg.strategy.weights.get("momentum", 0.25)       / total_w, 0.05, 0.60),
            _norm(cfg.strategy.weights.get("mean_reversion", 0.25) / total_w, 0.05, 0.60),
            _norm(cfg.strategy.weights.get("breakout", 0.25)       / total_w, 0.05, 0.60),
            _norm(cfg.strategy.weights.get("ai_prediction", 0.25)  / total_w, 0.05, 0.60),
            _norm(sl,                                  1.0,  3.5),
            _norm(tp_sl,                               1.2,  3.5),
            _norm(cfg.risk.risk_per_trade_pct,         0.005, 0.025),
        ]
        return cls(genes=genes)

    @classmethod
    def from_params(cls, params: Dict) -> "Chromosome":
        """
        Reconstruct a chromosome from a decoded params dict
        (e.g. loaded from optimized_params.json).
        """
        def _norm(val: float, lo: float, hi: float) -> float:
            return max(0.0, min(1.0, (val - lo) / (hi - lo) if hi > lo else 0.5))

        sl  = params.get("stop_loss_atr_mult", 2.0)
        tp  = params.get("take_profit_atr_mult", 4.0)
        w   = params.get("weights", {"momentum": 0.25, "mean_reversion": 0.25,
                                      "breakout": 0.25, "ai_prediction": 0.25})
        total_w = sum(w.values()) or 1.0

        genes = [
            _norm(params.get("rsi_period",         14),   7,    21),
            _norm(params.get("macd_fast",           12),   6,    16),
            _norm(params.get("macd_slow",           26) - params.get("macd_fast", 12), 8, 20),
            _norm(params.get("macd_signal",          9),   5,    13),
            _norm(params.get("atr_period",          14),   7,    21),
            _norm(params.get("bb_period",           20),   10,   30),
            _norm(params.get("bb_std",             2.0),   1.5,  3.0),
            _norm(params.get("volume_ma_period",    20),   10,   30),
            _norm(params.get("momentum_lookback",   20),   5,    40),
            _norm(params.get("zscore_threshold",   2.0),   1.0,  3.5),
            _norm(params.get("breakout_lookback",   20),   5,    40),
            _norm(params.get("signal_threshold",   0.5),   0.25, 0.75),
            _norm(w.get("momentum",       0.25) / total_w, 0.05, 0.60),
            _norm(w.get("mean_reversion", 0.25) / total_w, 0.05, 0.60),
            _norm(w.get("breakout",       0.25) / total_w, 0.05, 0.60),
            _norm(w.get("ai_prediction",  0.25) / total_w, 0.05, 0.60),
            _norm(sl,                              1.0,  3.5),
            _norm(tp / max(sl, 1e-6),              1.2,  3.5),
            _norm(params.get("risk_per_trade_pct", 0.01), 0.005, 0.025),
        ]
        return cls(genes=genes)

    # ------------------------------------------------------------------ #
    #  Decoding                                                            #
    # ------------------------------------------------------------------ #

    def decode(self) -> Dict:
        """
        Map normalised genes → actual parameter dict.

        Returns
        -------
        dict with keys matching config.settings sub-configs.
        """
        g = self.genes

        def _val(idx: int) -> float:
            gene = PARAM_SPACE[idx]
            raw = gene.lo + g[idx] * (gene.hi - gene.lo)
            return int(round(raw)) if gene.dtype is int else round(raw, 6)

        # ── Feature params ───────────────────────────────────────
        rsi_period       = _val(0)
        macd_fast        = _val(1)
        macd_slow        = macd_fast + _val(2)         # constraint: slow > fast
        macd_signal      = _val(3)
        atr_period       = _val(4)
        bb_period        = _val(5)
        bb_std           = _val(6)
        volume_ma_period = _val(7)

        # ── Strategy params ──────────────────────────────────────
        momentum_lookback = _val(8)
        zscore_threshold  = _val(9)
        breakout_lookback = _val(10)
        signal_threshold  = _val(11)

        # ── Strategy weights (L1-normalise) ──────────────────────
        raw_w  = [_val(12), _val(13), _val(14), _val(15)]
        total  = sum(raw_w) or 1.0
        weights = {
            "momentum":       round(raw_w[0] / total, 6),
            "mean_reversion": round(raw_w[1] / total, 6),
            "breakout":       round(raw_w[2] / total, 6),
            "ai_prediction":  round(raw_w[3] / total, 6),
        }

        # ── Risk params ──────────────────────────────────────────
        stop_loss_atr_mult   = _val(16)
        tp_sl_ratio          = _val(17)
        take_profit_atr_mult = round(stop_loss_atr_mult * tp_sl_ratio, 4)   # constraint: tp > sl
        risk_per_trade_pct   = _val(18)

        return {
            # feature engineering
            "rsi_period":            rsi_period,
            "macd_fast":             macd_fast,
            "macd_slow":             macd_slow,
            "macd_signal":           macd_signal,
            "atr_period":            atr_period,
            "bb_period":             bb_period,
            "bb_std":                bb_std,
            "volume_ma_period":      volume_ma_period,
            # strategy
            "momentum_lookback":     momentum_lookback,
            "zscore_threshold":      zscore_threshold,
            "breakout_lookback":     breakout_lookback,
            "signal_threshold":      signal_threshold,
            "weights":               weights,
            # risk
            "stop_loss_atr_mult":    stop_loss_atr_mult,
            "take_profit_atr_mult":  take_profit_atr_mult,
            "risk_per_trade_pct":    risk_per_trade_pct,
        }

    # ------------------------------------------------------------------ #
    #  Genetic operators                                                   #
    # ------------------------------------------------------------------ #

    def crossover(
        self,
        other: "Chromosome",
        rng: Optional[random.Random] = None,
    ) -> Tuple["Chromosome", "Chromosome"]:
        """
        Uniform crossover: each gene is independently drawn from either parent
        with equal probability.  Both offspring are returned.
        """
        _rng = rng or random
        c1, c2 = [], []
        for g1, g2 in zip(self.genes, other.genes):
            if _rng.random() < 0.5:
                c1.append(g1); c2.append(g2)
            else:
                c1.append(g2); c2.append(g1)
        return Chromosome(c1), Chromosome(c2)

    def mutate(
        self,
        sigma: float = 0.05,
        p_per_gene: float = 0.15,
        rng: Optional[random.Random] = None,
    ) -> "Chromosome":
        """
        Gaussian mutation in normalised [0, 1] space.
        Each gene is perturbed with probability `p_per_gene`.
        The result is clamped back to [0, 1].
        """
        _rng = rng or random
        new_genes = []
        for g in self.genes:
            if _rng.random() < p_per_gene:
                g = g + _rng.gauss(0.0, sigma)
                g = max(0.0, min(1.0, g))
            new_genes.append(g)
        return Chromosome(new_genes)

    def copy(self) -> "Chromosome":
        return Chromosome(list(self.genes))

    # ------------------------------------------------------------------ #
    #  Repr / comparison                                                   #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        p = self.decode()
        return (
            f"Chromosome("
            f"sl={p['stop_loss_atr_mult']:.2f}× "
            f"tp={p['take_profit_atr_mult']:.2f}× "
            f"risk={p['risk_per_trade_pct']*100:.1f}% "
            f"thresh={p['signal_threshold']:.2f} "
            f"rsi={p['rsi_period']} "
            f"w={[round(v,2) for v in p['weights'].values()]})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return NotImplemented
        return self.genes == other.genes

    def __hash__(self) -> int:
        return hash(tuple(round(g, 4) for g in self.genes))
