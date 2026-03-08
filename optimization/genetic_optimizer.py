"""
Genetic Algorithm optimiser for trading strategy parameters.

Algorithm
---------
1. Initialise a population of `pop_size` random chromosomes, seeding the
   first individual with the current default settings.
2. Evaluate fitness of every individual (parallel via multiprocessing.Pool
   or sequential when n_workers == 1).
3. Repeat for `n_generations`:
   a. Preserve `n_elite` best individuals unchanged (elitism).
   b. Fill the rest of the new population via tournament selection +
      uniform crossover + Gaussian mutation.
4. Early-stop if best fitness does not improve for `patience` generations.
5. Return the best chromosome found, full convergence history, and the
   final evaluated population.

Parallelism
-----------
Uses `multiprocessing.Pool` with a per-process initialiser that loads the
OHLCV data ONCE (not per evaluation call).  Falls back gracefully to
sequential evaluation if n_workers == 1 or multiprocessing fails.

On Windows the 'spawn' start method is used automatically; callers MUST
guard the entry point with `if __name__ == '__main__':`.
"""

from __future__ import annotations

import logging
import multiprocessing
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimization.chromosome import Chromosome, N_GENES
from optimization.fitness import (
    FitnessEvaluator,
    INVALID_FITNESS,
    _worker_eval,
    _worker_init,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────── convergence record ──────────────────


@dataclass
class GenerationStats:
    generation:    int
    best_fitness:  float
    mean_fitness:  float
    worst_fitness: float
    std_fitness:   float
    best_params:   Dict
    elapsed_sec:   float


@dataclass
class OptimisationResult:
    best_chromosome:     Chromosome
    best_fitness:        float
    best_params:         Dict
    history:             List[GenerationStats] = field(default_factory=list)
    final_population:    List[Chromosome]      = field(default_factory=list)
    final_fitnesses:     List[float]           = field(default_factory=list)
    train_metrics:       Dict                  = field(default_factory=dict)
    validation_metrics:  Dict                  = field(default_factory=dict)

    def convergence_df(self) -> pd.DataFrame:
        """Return a DataFrame suitable for plotting the convergence curve."""
        return pd.DataFrame([
            {
                "generation":    s.generation,
                "best_fitness":  s.best_fitness,
                "mean_fitness":  s.mean_fitness,
                "std_fitness":   s.std_fitness,
            }
            for s in self.history
        ])


# ─────────────────────── GA optimiser ────────────────────────


class GAOptimizer:
    """
    Genetic Algorithm optimiser for the 19-dimensional strategy parameter
    space defined in `optimization.chromosome`.

    Parameters
    ----------
    pop_size      : number of individuals per generation (default 50)
    n_generations : maximum number of generations (default 30)
    n_elite       : number of best individuals preserved unchanged (default 5)
    tournament_k  : tournament selection pool size (default 3)
    p_crossover   : probability two selected parents undergo crossover (0-1)
    mutation_sigma: std-dev of per-gene Gaussian perturbation in [0,1]
    p_mutate_gene : per-gene mutation probability
    patience      : early-stop if no improvement for this many generations
    n_workers     : worker processes for parallel evaluation (1 = sequential)
    seed          : random seed for reproducibility (None = random)
    """

    def __init__(
        self,
        pop_size:       int   = 50,
        n_generations:  int   = 30,
        n_elite:        int   = 5,
        tournament_k:   int   = 3,
        p_crossover:    float = 0.85,
        mutation_sigma: float = 0.05,
        p_mutate_gene:  float = 0.15,
        patience:       int   = 10,
        n_workers:      int   = 1,
        seed:           Optional[int] = None,
    ) -> None:
        self.pop_size       = pop_size
        self.n_generations  = n_generations
        self.n_elite        = n_elite
        self.tournament_k   = tournament_k
        self.p_crossover    = p_crossover
        self.mutation_sigma = mutation_sigma
        self.p_mutate_gene  = p_mutate_gene
        self.patience       = patience
        self.n_workers      = n_workers
        self.seed           = seed
        self._rng           = random.Random(seed)

        logger.info(
            "GAOptimizer config: pop=%d | gen=%d | elite=%d | workers=%d | "
            "sigma=%.3f | p_mut=%.2f | patience=%d",
            pop_size, n_generations, n_elite, n_workers,
            mutation_sigma, p_mutate_gene, patience,
        )

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(
        self,
        df_dict: Dict[str, pd.DataFrame],
        symbols: List[str],
        initial_capital: float = 100_000.0,
        commission_pct:  float = 0.001,
        slippage_pct:    float = 0.0005,
        progress_cb: Optional[Callable[[GenerationStats], None]] = None,
    ) -> OptimisationResult:
        """
        Run the full GA optimisation loop.

        Parameters
        ----------
        df_dict         : {symbol: ohlcv_df} — TRAINING data only
        symbols         : ordered list of symbols to optimise against
        initial_capital : simulation starting capital
        commission_pct  : per-trade commission rate
        slippage_pct    : per-trade slippage rate
        progress_cb     : optional callback called with GenerationStats each gen

        Returns
        -------
        OptimisationResult with best chromosome, history, and metrics
        """
        t_start = time.time()
        logger.info("=" * 60)
        logger.info("GA optimisation start  |  symbols=%s", symbols)
        logger.info("=" * 60)

        # ── Initialise population ─────────────────────────────────
        population = self._init_population()

        # ── Evaluator (used for final reporting) ──────────────────
        evaluator = FitnessEvaluator(
            df_dict, symbols, initial_capital, commission_pct, slippage_pct
        )

        # ── Evolution loop ────────────────────────────────────────
        history:      List[GenerationStats] = []
        best_ever:    float                 = INVALID_FITNESS - 1
        best_chrom:   Chromosome            = population[0]
        no_improve_n: int                   = 0

        fitnesses: List[float] = self._evaluate_population(
            population, df_dict, symbols, initial_capital, commission_pct, slippage_pct
        )

        for gen in range(self.n_generations):
            gen_t = time.time()

            # Track best
            best_idx = int(np.argmax(fitnesses))
            gen_best = fitnesses[best_idx]

            if gen_best > best_ever:
                best_ever  = gen_best
                best_chrom = population[best_idx].copy()
                no_improve_n = 0
            else:
                no_improve_n += 1

            # Stats
            fit_arr = np.array(fitnesses, dtype=float)
            stats = GenerationStats(
                generation    = gen,
                best_fitness  = float(gen_best),
                mean_fitness  = float(fit_arr.mean()),
                worst_fitness = float(fit_arr.min()),
                std_fitness   = float(fit_arr.std()),
                best_params   = population[best_idx].decode(),
                elapsed_sec   = time.time() - gen_t,
            )
            history.append(stats)

            logger.info(
                "Gen %3d/%d | best=%.4f | mean=%.4f | std=%.4f | "
                "no_improve=%d | %.1fs",
                gen, self.n_generations,
                stats.best_fitness, stats.mean_fitness, stats.std_fitness,
                no_improve_n, time.time() - t_start,
            )

            if progress_cb:
                progress_cb(stats)

            # ── Early stopping ────────────────────────────────────
            if no_improve_n >= self.patience:
                logger.info(
                    "Early stopping: no improvement for %d generations", self.patience
                )
                break

            # ── Build next generation ─────────────────────────────
            elites     = self._get_elites(population, fitnesses)
            new_pop    = list(elites)

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)

                if self._rng.random() < self.p_crossover:
                    c1, c2 = p1.crossover(p2, rng=self._rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = c1.mutate(self.mutation_sigma, self.p_mutate_gene, self._rng)
                c2 = c2.mutate(self.mutation_sigma, self.p_mutate_gene, self._rng)
                new_pop.extend([c1, c2])

            population = new_pop[: self.pop_size]

            # ── Re-evaluate new population ────────────────────────
            fitnesses = self._evaluate_population(
                population, df_dict, symbols,
                initial_capital, commission_pct, slippage_pct,
            )

        # ── Retrieve final best ───────────────────────────────────
        best_idx     = int(np.argmax(fitnesses))
        final_best_f = fitnesses[best_idx]
        if final_best_f > best_ever:
            best_ever  = final_best_f
            best_chrom = population[best_idx].copy()

        best_params = best_chrom.decode()
        logger.info("=" * 60)
        logger.info(
            "Optimisation complete in %.1f s  |  best_fitness=%.4f",
            time.time() - t_start, best_ever,
        )
        logger.info("Best params: %s", best_chrom)
        logger.info("=" * 60)

        # Full metrics on training data for the best chromosome
        _, train_metrics = evaluator.evaluate_with_metrics(best_chrom)

        return OptimisationResult(
            best_chromosome  = best_chrom,
            best_fitness     = best_ever,
            best_params      = best_params,
            history          = history,
            final_population = population,
            final_fitnesses  = fitnesses,
            train_metrics    = train_metrics,
        )

    # ------------------------------------------------------------------ #
    #  Population initialisation                                           #
    # ------------------------------------------------------------------ #

    def _init_population(self) -> List[Chromosome]:
        """
        Seed the population with:
          - 1 chromosome from current default settings (exploitation)
          - (pop_size - 1) random chromosomes (exploration)
        """
        pop: List[Chromosome] = [Chromosome.from_default_settings()]
        while len(pop) < self.pop_size:
            pop.append(Chromosome.random(rng=self._rng))
        logger.info("Population initialised: %d individuals", self.pop_size)
        return pop

    # ------------------------------------------------------------------ #
    #  Fitness evaluation (sequential or parallel)                        #
    # ------------------------------------------------------------------ #

    def _evaluate_population(
        self,
        population:      List[Chromosome],
        df_dict:         Dict[str, pd.DataFrame],
        symbols:         List[str],
        initial_capital: float,
        commission_pct:  float,
        slippage_pct:    float,
    ) -> List[float]:
        if self.n_workers > 1:
            return self._evaluate_parallel(
                population, df_dict, symbols,
                initial_capital, commission_pct, slippage_pct,
            )
        return self._evaluate_sequential(
            population, df_dict, symbols,
            initial_capital, commission_pct, slippage_pct,
        )

    def _evaluate_sequential(
        self,
        population:      List[Chromosome],
        df_dict:         Dict[str, pd.DataFrame],
        symbols:         List[str],
        initial_capital: float,
        commission_pct:  float,
        slippage_pct:    float,
    ) -> List[float]:
        evaluator = FitnessEvaluator(
            df_dict, symbols, initial_capital, commission_pct, slippage_pct
        )
        fitnesses = []
        for chrom in population:
            fitnesses.append(evaluator.evaluate(chrom))
        return fitnesses

    def _evaluate_parallel(
        self,
        population:      List[Chromosome],
        df_dict:         Dict[str, pd.DataFrame],
        symbols:         List[str],
        initial_capital: float,
        commission_pct:  float,
        slippage_pct:    float,
    ) -> List[float]:
        """
        Evaluate the population using a multiprocessing.Pool.
        Each worker receives the DataFrames once via the initializer and
        only the gene lists (small) per evaluation call.
        """
        genes_list = [c.genes for c in population]
        n_workers  = min(self.n_workers, multiprocessing.cpu_count(), len(population))

        try:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                processes   = n_workers,
                initializer = _worker_init,
                initargs    = (df_dict, symbols, initial_capital,
                               commission_pct, slippage_pct),
            ) as pool:
                fitnesses = pool.map(_worker_eval, genes_list)
            return fitnesses

        except Exception as exc:
            logger.warning(
                "Parallel evaluation failed (%s) – falling back to sequential", exc
            )
            return self._evaluate_sequential(
                population, df_dict, symbols,
                initial_capital, commission_pct, slippage_pct,
            )

    # ------------------------------------------------------------------ #
    #  Selection                                                           #
    # ------------------------------------------------------------------ #

    def _tournament_select(
        self, population: List[Chromosome], fitnesses: List[float]
    ) -> Chromosome:
        """
        Tournament selection: draw `tournament_k` candidates at random and
        return the one with the highest fitness.
        """
        k         = min(self.tournament_k, len(population))
        candidates = self._rng.sample(range(len(population)), k)
        winner     = max(candidates, key=lambda i: fitnesses[i])
        return population[winner].copy()

    def _get_elites(
        self, population: List[Chromosome], fitnesses: List[float]
    ) -> List[Chromosome]:
        """Return the `n_elite` best individuals (copied)."""
        n = min(self.n_elite, len(population))
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        return [population[i].copy() for i in ranked[:n]]

    # ------------------------------------------------------------------ #
    #  Diversity helpers                                                   #
    # ------------------------------------------------------------------ #

    def population_diversity(self, population: List[Chromosome]) -> float:
        """
        Return the mean pairwise L2 distance in gene space.
        A value close to 0 means the population has converged.
        """
        genes_arr = np.array([c.genes for c in population])
        dists = []
        for i in range(len(genes_arr)):
            for j in range(i + 1, len(genes_arr)):
                dists.append(float(np.linalg.norm(genes_arr[i] - genes_arr[j])))
        return float(np.mean(dists)) if dists else 0.0


# ─────────────────────── adaptive mutation ───────────────────


class AdaptiveGAOptimizer(GAOptimizer):
    """
    Extends `GAOptimizer` with adaptive mutation rate.

    When population diversity falls below `div_threshold`, mutation sigma
    is temporarily increased to `sigma_boost` to reintroduce exploration.
    """

    def __init__(
        self,
        div_threshold: float = 0.3,
        sigma_boost:   float = 0.12,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.div_threshold = div_threshold
        self.sigma_boost   = sigma_boost
        self._base_sigma   = self.mutation_sigma

    def run(self, *args, **kwargs) -> OptimisationResult:
        # Wrap parent run so we can inject a progress callback that
        # adjusts mutation sigma based on population diversity.
        # Since the parent loop is inside `run`, we hook via the
        # pre/post generation evaluation steps by subclassing _evolve.
        return super().run(*args, **kwargs)

    def _evaluate_population(self, population, *args, **kwargs):
        """Override to check diversity and adapt sigma after each evaluation."""
        fitnesses = super()._evaluate_population(population, *args, **kwargs)
        diversity  = self.population_diversity(population)
        if diversity < self.div_threshold:
            self.mutation_sigma = self.sigma_boost
            logger.debug(
                "Low diversity (%.3f < %.3f) → boosted sigma to %.3f",
                diversity, self.div_threshold, self.sigma_boost,
            )
        else:
            self.mutation_sigma = self._base_sigma
        return fitnesses
