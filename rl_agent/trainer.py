"""
PPO training loop.

Architecture of the training cycle
────────────────────────────────────
1. Create a MultiAssetEnv wrapping all training symbols.
2. Reset the environment; collect n_steps transitions using the current policy.
3. Bootstrap the final value for GAE computation.
4. Call agent.update() to run n_epochs of PPO-Clip gradient steps.
5. Every eval_interval rollouts, evaluate on a held-out evaluation set and
   track the best model (by mean episode return).
6. Save the best checkpoint and the final model; write a JSON metrics log.

Usage
─────
    trainer = PPOTrainer(df_dict={"BTC/USDT": df_btc, "ETH/USDT": df_eth})
    result  = trainer.train()
    # or from CLI:  python rl_agent/trainer.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config.settings import settings
from rl_agent.ppo_agent import PPOAgent
from rl_agent.trading_env import MultiAssetEnv, TradingEnv, OBS_DIM, ACTION_DIM
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────── result dataclass ────────────────────

@dataclass
class TrainingResult:
    total_timesteps:      int
    total_episodes:       int
    best_mean_return:     float
    final_mean_return:    float
    training_time_sec:    float
    episode_returns:      List[float]     = field(default_factory=list)
    eval_returns:         List[float]     = field(default_factory=list)
    policy_losses:        List[float]     = field(default_factory=list)
    value_losses:         List[float]     = field(default_factory=list)
    entropies:            List[float]     = field(default_factory=list)
    approx_kls:           List[float]     = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_timesteps":   self.total_timesteps,
            "total_episodes":    self.total_episodes,
            "best_mean_return":  round(self.best_mean_return, 4),
            "final_mean_return": round(self.final_mean_return, 4),
            "training_time_sec": round(self.training_time_sec, 1),
        }


# ─────────────────────── evaluation helper ───────────────────

def evaluate_agent(
    agent:          PPOAgent,
    eval_env:       MultiAssetEnv,
    n_episodes:     int   = 10,
    deterministic:  bool  = True,
) -> Tuple[float, float, float]:
    """
    Roll out `n_episodes` episodes with the agent in evaluation mode.

    Returns
    -------
    mean_return, std_return, mean_final_portfolio_pct
    """
    ep_returns    = []
    ep_portfolios = []

    for _ in range(n_episodes):
        obs, _  = eval_env.reset()
        ep_ret  = 0.0
        done    = False

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_ret += reward
            done    = terminated or truncated

        ep_returns.append(ep_ret)
        ep_portfolios.append(
            eval_env._active_env.total_return_pct
            if eval_env._active_env else 0.0
        )

    return (
        float(np.mean(ep_returns)),
        float(np.std(ep_returns)),
        float(np.mean(ep_portfolios)),
    )


# ─────────────────────── trainer ─────────────────────────────

class PPOTrainer:
    """
    Orchestrates the PPO training loop over a set of trading environments.

    Parameters
    ----------
    df_dict          : {symbol: ohlcv_dataframe} — use only TRAIN data
    eval_df_dict     : optional separate evaluation set (default = df_dict)
    total_timesteps  : total environment steps to collect
    eval_interval    : evaluate every this many rollouts (PPO updates)
    checkpoint_dir   : directory to save model checkpoints
    rl_config        : RLConfig from settings (or defaults below)
    """

    DEFAULT_MODEL_PATH = settings.model_dir / "ppo_agent.pt"
    DEFAULT_BEST_PATH  = settings.model_dir / "ppo_agent_best.pt"
    METRICS_PATH       = settings.model_dir / "ppo_training_metrics.json"

    def __init__(
        self,
        df_dict:         Dict[str, pd.DataFrame],
        eval_df_dict:    Optional[Dict[str, pd.DataFrame]] = None,
        total_timesteps: int   = 500_000,
        eval_interval:   int   = 10,
        episode_length:  int   = 252,
        initial_capital: float = 100_000.0,
        # PPO hyper-parameters (override via settings.rl if present)
        lr:              float = 3e-4,
        n_steps:         int   = 2048,
        batch_size:      int   = 64,
        n_epochs:        int   = 10,
        hidden_dim:      int   = 256,
        use_lstm:        bool  = False,
        gamma:           float = 0.99,
        gae_lambda:      float = 0.95,
        clip_range:      float = 0.20,
        ent_coef:        float = 0.01,
        vf_coef:         float = 0.50,
        target_kl:       float = 0.015,
        seed:            Optional[int] = None,
        device:          str   = "auto",
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required.  Run: pip install torch")

        # Override from settings.rl if it exists
        rl_cfg = getattr(settings, "rl", None)
        if rl_cfg is not None:
            total_timesteps = getattr(rl_cfg, "total_timesteps", total_timesteps)
            lr              = getattr(rl_cfg, "lr",              lr)
            n_steps         = getattr(rl_cfg, "n_steps",         n_steps)
            batch_size      = getattr(rl_cfg, "batch_size",      batch_size)
            n_epochs        = getattr(rl_cfg, "n_epochs",        n_epochs)
            hidden_dim      = getattr(rl_cfg, "hidden_dim",      hidden_dim)
            gamma           = getattr(rl_cfg, "gamma",           gamma)
            gae_lambda      = getattr(rl_cfg, "gae_lambda",      gae_lambda)
            clip_range      = getattr(rl_cfg, "clip_range",      clip_range)
            ent_coef        = getattr(rl_cfg, "ent_coef",        ent_coef)
            episode_length  = getattr(rl_cfg, "episode_length",  episode_length)

        self.total_timesteps = total_timesteps
        self.eval_interval   = eval_interval
        self.initial_capital = initial_capital

        # Environments
        env_kwargs = dict(
            initial_capital = initial_capital,
            episode_length  = episode_length,
            seed            = seed,
        )
        self.train_env = MultiAssetEnv(df_dict, **env_kwargs)
        self.eval_env  = MultiAssetEnv(eval_df_dict or df_dict, **env_kwargs)

        # Agent
        self.agent = PPOAgent(
            obs_dim       = OBS_DIM,
            action_dim    = ACTION_DIM,
            hidden_dim    = hidden_dim,
            use_lstm      = use_lstm,
            lr            = lr,
            n_steps       = n_steps,
            batch_size    = batch_size,
            n_epochs      = n_epochs,
            gamma         = gamma,
            gae_lambda    = gae_lambda,
            clip_range    = clip_range,
            ent_coef      = ent_coef,
            vf_coef       = vf_coef,
            target_kl     = target_kl,
            device        = device,
        )

        logger.info(
            "PPOTrainer ready | timesteps=%d | symbols=%s",
            total_timesteps,
            list(self.train_env.envs.keys()),
        )

    # ------------------------------------------------------------------ #
    #  Main training loop                                                  #
    # ------------------------------------------------------------------ #

    def train(self) -> TrainingResult:
        """
        Run the full PPO training loop.

        Returns a `TrainingResult` with training statistics.
        """
        t_start       = time.time()
        timestep      = 0
        rollout_count = 0
        episode       = 0
        best_eval_ret = -np.inf

        episode_returns: List[float] = []
        eval_returns:    List[float] = []
        policy_losses:   List[float] = []
        value_losses:    List[float] = []
        entropies:       List[float] = []
        approx_kls:      List[float] = []

        # Initialise environment
        obs, _ = self.train_env.reset()
        ep_ret = 0.0

        logger.info("=" * 60)
        logger.info("PPO Training started | total_timesteps=%d", self.total_timesteps)
        logger.info("=" * 60)

        while timestep < self.total_timesteps:
            # ── Collect one rollout ───────────────────────────────
            self.agent.buffer.reset()
            self.agent.network.eval()

            for step in range(self.agent.n_steps):
                action, log_prob, value = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.train_env.step(action)
                done = terminated or truncated

                self.agent.buffer.add(obs, action, reward, done, value, log_prob)

                obs     = next_obs
                ep_ret += reward
                timestep += 1

                if done:
                    episode += 1
                    episode_returns.append(ep_ret)
                    ep_ret  = 0.0
                    obs, _  = self.train_env.reset()

                if timestep >= self.total_timesteps:
                    break

            # Bootstrap final value for GAE
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                _, _, last_value = self.agent.network.get_action(obs_t)
                last_value = float(last_value.item())

            self.agent.buffer.compute_returns_and_advantages(
                last_value = last_value,
                last_done  = done,
                gamma      = self.agent.gamma,
                gae_lambda = self.agent.gae_lambda,
            )

            # ── PPO update ────────────────────────────────────────
            self.agent.network.train()
            update_metrics = self.agent.update()

            policy_losses.append(update_metrics.get("policy_loss", 0.0))
            value_losses.append(update_metrics.get("value_loss",  0.0))
            entropies.append(update_metrics.get("entropy",       0.0))
            approx_kls.append(update_metrics.get("approx_kl",    0.0))
            rollout_count += 1

            # ── Logging ───────────────────────────────────────────
            mean_ep_ret = float(np.mean(episode_returns[-20:])) if episode_returns else 0.0
            logger.info(
                "Rollout %4d | steps=%7d | ep_return(20avg)=%7.2f | "
                "policy_loss=%6.4f | value_loss=%6.4f | entropy=%5.3f | "
                "kl=%5.4f | lr=%.2e",
                rollout_count, timestep, mean_ep_ret,
                update_metrics.get("policy_loss", 0),
                update_metrics.get("value_loss",  0),
                update_metrics.get("entropy",     0),
                update_metrics.get("approx_kl",   0),
                self.agent.current_lr,
            )

            # ── Periodic evaluation ───────────────────────────────
            if rollout_count % self.eval_interval == 0:
                self.agent.network.eval()
                mean_r, std_r, mean_port = evaluate_agent(
                    self.agent, self.eval_env, n_episodes=10
                )
                eval_returns.append(mean_r)
                logger.info(
                    "  EVAL | mean_return=%.2f ± %.2f | "
                    "portfolio_return=%.2f%%",
                    mean_r, std_r, mean_port,
                )

                # Save best model
                if mean_r > best_eval_ret:
                    best_eval_ret = mean_r
                    self.agent.save(str(self.DEFAULT_BEST_PATH))
                    logger.info("  → New best model saved (%.4f)", best_eval_ret)

        # ── Final save ────────────────────────────────────────────
        self.agent.save(str(self.DEFAULT_MODEL_PATH))

        # Final evaluation
        self.agent.network.eval()
        final_mean_r, _, _ = evaluate_agent(
            self.agent, self.eval_env, n_episodes=20
        )

        elapsed = time.time() - t_start
        logger.info("=" * 60)
        logger.info(
            "Training complete in %.1f s | best_eval=%.4f | final_eval=%.4f",
            elapsed, best_eval_ret, final_mean_r,
        )

        result = TrainingResult(
            total_timesteps   = timestep,
            total_episodes    = episode,
            best_mean_return  = best_eval_ret,
            final_mean_return = final_mean_r,
            training_time_sec = elapsed,
            episode_returns   = episode_returns,
            eval_returns      = eval_returns,
            policy_losses     = policy_losses,
            value_losses      = value_losses,
            entropies         = entropies,
            approx_kls        = approx_kls,
        )

        # Persist metrics
        self._save_metrics(result)
        return result

    # ------------------------------------------------------------------ #
    #  Persistence helpers                                                 #
    # ------------------------------------------------------------------ #

    def _save_metrics(self, result: TrainingResult) -> None:
        payload = {
            "summary": result.to_dict(),
            "episode_returns_last50": result.episode_returns[-50:],
            "eval_returns":           result.eval_returns,
            "policy_losses_last50":   result.policy_losses[-50:],
            "value_losses_last50":    result.value_losses[-50:],
            "entropies_last50":       result.entropies[-50:],
        }
        self.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.METRICS_PATH.write_text(json.dumps(payload, indent=2))
        logger.info("Training metrics saved → %s", self.METRICS_PATH)

    def load_best(self) -> bool:
        """Load the best checkpoint saved during training."""
        if self.DEFAULT_BEST_PATH.exists():
            self.agent.load(str(self.DEFAULT_BEST_PATH))
            return True
        return False


# ─────────────────────── CLI entry point ─────────────────────

def main() -> None:
    import argparse
    import sys

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    parser = argparse.ArgumentParser(description="Train PPO trading agent")
    parser.add_argument("--symbols",     nargs="+", default=None)
    parser.add_argument("--timesteps",   type=int,  default=500_000)
    parser.add_argument("--lr",          type=float,default=3e-4)
    parser.add_argument("--n-steps",     type=int,  default=2048)
    parser.add_argument("--batch-size",  type=int,  default=64)
    parser.add_argument("--n-epochs",    type=int,  default=10)
    parser.add_argument("--hidden-dim",  type=int,  default=256)
    parser.add_argument("--episode-len", type=int,  default=252)
    parser.add_argument("--eval-interval",type=int, default=10)
    parser.add_argument("--train-frac",  type=float,default=0.70)
    parser.add_argument("--lstm",        action="store_true")
    parser.add_argument("--seed",        type=int,  default=None)
    parser.add_argument("--device",      type=str,  default="auto")
    args = parser.parse_args()

    from data_pipeline.storage import DataStorage
    symbols   = args.symbols or settings.trading.symbols
    storage   = DataStorage()
    all_data: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        df = storage.load_ohlcv(sym, settings.trading.timeframe)
        if not df.empty:
            all_data[sym] = df
            logger.info("Loaded %d candles for %s", len(df), sym)

    if not all_data:
        logger.error("No data available.  Run `python main.py --mode fetch` first.")
        sys.exit(1)

    # Train / validation split
    train_data, val_data = {}, {}
    for sym, df in all_data.items():
        n = int(len(df) * args.train_frac)
        train_data[sym] = df.iloc[:n]
        val_data[sym]   = df.iloc[n:]

    trainer = PPOTrainer(
        df_dict         = train_data,
        eval_df_dict    = val_data,
        total_timesteps = args.timesteps,
        eval_interval   = args.eval_interval,
        episode_length  = args.episode_len,
        lr              = args.lr,
        n_steps         = args.n_steps,
        batch_size      = args.batch_size,
        n_epochs        = args.n_epochs,
        hidden_dim      = args.hidden_dim,
        use_lstm        = args.lstm,
        seed            = args.seed,
        device          = args.device,
    )
    result = trainer.train()
    print("\nTraining summary:")
    for k, v in result.to_dict().items():
        print(f"  {k:<28} {v}")


if __name__ == "__main__":
    main()
