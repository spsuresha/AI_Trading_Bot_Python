"""
Proximal Policy Optimisation (PPO) — from scratch implementation.

Key components
──────────────
RolloutBuffer  – stores a fixed-length trajectory; computes GAE advantages
PPOAgent       – wraps ActorCritic; implements the PPO-Clip update loop

PPO-Clip objective
──────────────────
  L_CLIP = E[ min( r(θ) A, clip(r(θ), 1−ε, 1+ε) A ) ]

  where  r(θ) = π_θ(a|s) / π_θ_old(a|s)   (probability ratio)
         A    = GAE advantage estimate

  Total loss = −L_CLIP + c_v × L_VF − c_e × H
  where L_VF = MSE(V(s), returns) and H = policy entropy.

References
──────────
  Schulman et al. 2017  https://arxiv.org/abs/1707.06347
  GAE: Schulman et al. 2015  https://arxiv.org/abs/1506.02438
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LinearLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from rl_agent.networks import ActorCritic, LSTMActorCritic
from utils.logger import get_logger

logger = get_logger(__name__)


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.  Install with:  pip install torch")


# ─────────────────────── rollout buffer ──────────────────────


class RolloutBuffer:
    """
    Fixed-capacity ring buffer for on-policy PPO rollouts.

    Stores one complete rollout of `buffer_size` steps, then computes
    Generalised Advantage Estimates (GAE) before the PPO update.
    """

    def __init__(self, buffer_size: int, obs_dim: int) -> None:
        self.buffer_size = buffer_size
        self.obs_dim     = obs_dim
        self._pos        = 0
        self._full       = False
        self._allocate()

    def _allocate(self) -> None:
        n = self.buffer_size
        self.observations = np.zeros((n, self.obs_dim), dtype=np.float32)
        self.actions      = np.zeros(n,                dtype=np.int64)
        self.rewards      = np.zeros(n,                dtype=np.float32)
        self.dones        = np.zeros(n,                dtype=np.float32)
        self.values       = np.zeros(n,                dtype=np.float32)
        self.log_probs    = np.zeros(n,                dtype=np.float32)
        self.advantages   = np.zeros(n,                dtype=np.float32)
        self.returns      = np.zeros(n,                dtype=np.float32)

    def reset(self) -> None:
        self._pos  = 0
        self._full = False

    def add(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        done:     bool,
        value:    float,
        log_prob: float,
    ) -> None:
        self.observations[self._pos] = obs
        self.actions  [self._pos]    = action
        self.rewards  [self._pos]    = reward
        self.dones    [self._pos]    = float(done)
        self.values   [self._pos]    = value
        self.log_probs[self._pos]    = log_prob
        self._pos += 1
        if self._pos >= self.buffer_size:
            self._full = True

    @property
    def is_full(self) -> bool:
        return self._full

    # ------------------------------------------------------------------ #
    #  GAE computation                                                     #
    # ------------------------------------------------------------------ #

    def compute_returns_and_advantages(
        self,
        last_value:  float,
        last_done:   bool,
        gamma:       float = 0.99,
        gae_lambda:  float = 0.95,
    ) -> None:
        """
        Compute GAE advantages and discounted returns IN-PLACE.

        Parameters
        ----------
        last_value : V(s_{T+1}) — the bootstrapped value of the state after
                     the last collected transition.
        last_done  : whether the final state was terminal.
        """
        last_gae   = 0.0
        next_val   = last_value
        next_notdone = 1.0 - float(last_done)

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = next_notdone
                next_value        = next_val
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value        = self.values[t + 1]

            delta    = (self.rewards[t]
                        + gamma * next_value * next_non_terminal
                        - self.values[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------ #
    #  Mini-batch generator                                                #
    # ------------------------------------------------------------------ #

    def get_batches(
        self,
        batch_size: int,
        device:     str = "cpu",
    ) -> Generator[Tuple, None, None]:
        """
        Yield shuffled mini-batches as PyTorch tensors.
        Advantages are normalised to zero-mean / unit-variance per batch.
        """
        indices = np.random.permutation(self.buffer_size)
        # Normalise advantages globally (not per-batch for stability)
        adv = self.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, self.buffer_size, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                torch.FloatTensor(self.observations[idx]).to(device),
                torch.LongTensor (self.actions      [idx]).to(device),
                torch.FloatTensor(self.log_probs    [idx]).to(device),
                torch.FloatTensor(adv               [idx]).to(device),
                torch.FloatTensor(self.returns      [idx]).to(device),
            )


# ─────────────────────── PPO agent ───────────────────────────


class PPOAgent:
    """
    Full PPO-Clip agent with:
      • Shared Actor-Critic network (MLP or LSTM)
      • Rollout buffer with GAE
      • Clipped policy-gradient loss
      • Clipped value-function loss
      • Entropy regularisation
      • Gradient clipping
      • Optional linear learning-rate annealing
    """

    def __init__(
        self,
        obs_dim:           int,
        action_dim:        int   = 3,
        hidden_dim:        int   = 256,
        use_lstm:          bool  = False,
        lr:                float = 3e-4,
        n_steps:           int   = 2048,   # rollout length
        batch_size:        int   = 64,
        n_epochs:          int   = 10,
        gamma:             float = 0.99,
        gae_lambda:        float = 0.95,
        clip_range:        float = 0.20,
        clip_range_vf:     Optional[float] = 0.20,
        ent_coef:          float = 0.01,
        vf_coef:           float = 0.50,
        max_grad_norm:     float = 0.50,
        target_kl:         Optional[float] = 0.015,
        lr_anneal_steps:   int   = 0,       # 0 = no annealing
        device:            str   = "auto",
    ) -> None:
        _require_torch()

        self.obs_dim       = obs_dim
        self.action_dim    = action_dim
        self.n_steps       = n_steps
        self.batch_size    = batch_size
        self.n_epochs      = n_epochs
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_range    = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl     = target_kl

        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Network
        if use_lstm:
            self.network = LSTMActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.network = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)

        # Optimiser
        self.optimizer = Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Optional LR scheduler
        self._scheduler = None
        if lr_anneal_steps > 0:
            self._scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=lr_anneal_steps,
            )

        # Rollout buffer
        self.buffer = RolloutBuffer(n_steps, obs_dim)

        # Stats tracking
        self._n_updates = 0

        logger.info(
            "PPOAgent | obs=%d | actions=%d | device=%s | "
            "lr=%.0e | n_steps=%d | batch=%d | epochs=%d",
            obs_dim, action_dim, self.device,
            lr, n_steps, batch_size, n_epochs,
        )

    # ------------------------------------------------------------------ #
    #  Action selection (called during rollout collection)                 #
    # ------------------------------------------------------------------ #

    def select_action(
        self,
        obs:           np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Parameters
        ----------
        obs : (obs_dim,) float32 array

        Returns
        -------
        action   : int
        log_prob : float
        value    : float
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_t, log_prob_t, value_t = self.network.get_action(
            obs_t, deterministic=deterministic
        )
        return (
            int(action_t.item()),
            float(log_prob_t.item()),
            float(value_t.item()),
        )

    # ------------------------------------------------------------------ #
    #  PPO update                                                          #
    # ------------------------------------------------------------------ #

    def update(self) -> Dict[str, float]:
        """
        Run `n_epochs` passes of PPO update over the current rollout buffer.

        Returns a dict of average training metrics for logging.
        """
        _require_torch()
        metrics: Dict[str, List[float]] = defaultdict(list)
        early_stop = False

        for epoch in range(self.n_epochs):
            if early_stop:
                break

            for batch in self.buffer.get_batches(self.batch_size, self.device):
                obs_b, act_b, old_lp_b, adv_b, ret_b = batch

                # Re-evaluate actions under current policy
                new_lp, values, entropy = self.network.evaluate_actions(obs_b, act_b)

                # ── Policy (PPO-Clip) loss ────────────────────────
                ratio      = torch.exp(new_lp - old_lp_b)
                loss_clip1 = -adv_b * ratio
                loss_clip2 = -adv_b * torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                policy_loss = torch.max(loss_clip1, loss_clip2).mean()

                # ── Value (MSE) loss ──────────────────────────────
                if self.clip_range_vf is not None:
                    # Clipped value update (Schulman et al. implementation trick)
                    values_clipped = torch.clamp(
                        values,
                        ret_b - self.clip_range_vf,
                        ret_b + self.clip_range_vf,
                    )
                    value_loss = torch.max(
                        F.mse_loss(values, ret_b),
                        F.mse_loss(values_clipped, ret_b),
                    )
                else:
                    value_loss = F.mse_loss(values, ret_b)

                # ── Entropy bonus ─────────────────────────────────
                entropy_loss = -entropy.mean()

                # ── Total loss ────────────────────────────────────
                loss = (
                    policy_loss
                    + self.vf_coef   * value_loss
                    + self.ent_coef  * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (new_lp - old_lp_b)).mean().item()
                    clip_frac  = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_frac"].append(clip_frac)
                metrics["total_loss"].append(loss.item())

            # KL early-stop
            if self.target_kl is not None:
                mean_kl = float(np.mean(metrics["approx_kl"]))
                if mean_kl > 1.5 * self.target_kl:
                    logger.debug(
                        "Early-stop at epoch %d: KL %.4f > %.4f",
                        epoch, mean_kl, 1.5 * self.target_kl,
                    )
                    early_stop = True

        if self._scheduler is not None:
            self._scheduler.step()

        self._n_updates += 1
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save network weights and optimiser state."""
        _require_torch()
        torch.save({
            "network_state":    self.network.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "n_updates":        self._n_updates,
            "obs_dim":          self.obs_dim,
            "action_dim":       self.action_dim,
        }, path)
        logger.info("PPOAgent saved → %s", path)

    def load(self, path: str) -> None:
        """Load network weights and optimiser state."""
        _require_torch()
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._n_updates = checkpoint.get("n_updates", 0)
        self.network.eval()
        logger.info("PPOAgent loaded from %s (updates=%d)", path, self._n_updates)

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])
