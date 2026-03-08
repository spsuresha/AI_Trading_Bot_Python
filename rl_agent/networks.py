"""
Actor-Critic neural networks for PPO.

Architecture
────────────
Input (obs_dim,)
  └─ Shared trunk: Linear → Tanh → Linear → Tanh   (orthogonal init, √2 gain)
       ├─ Policy head: Linear → (raw logits for Categorical distribution)
       └─ Value head:  Linear → scalar                (orthogonal init, 1.0 gain)

Policy-head uses gain=0.01 (near-uniform initial action distribution).

Both heads are separated from the trunk so gradient magnitudes from the
policy and value losses don't interfere with the shared representation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the RL agent.  "
            "Install it with:  pip install torch"
        )


def _ortho_init(layer: "nn.Linear", gain: float = np.sqrt(2)) -> "nn.Linear":
    """Apply orthogonal initialisation to a Linear layer."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


# ─────────────────────── shared trunk ────────────────────────

class MLP(nn.Module):
    """
    Two-layer MLP with Tanh activations and orthogonal weight init.
    Used as the shared feature extractor in ActorCritic.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        _check_torch()
        super().__init__()
        self.net = nn.Sequential(
            _ortho_init(nn.Linear(in_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
            _ortho_init(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ─────────────────────── actor-critic ────────────────────────

class ActorCritic(nn.Module):
    """
    Shared Actor-Critic network for PPO.

    ``get_action`` is used during rollout collection.
    ``evaluate_actions`` is used during the PPO update step.
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        _check_torch()
        super().__init__()

        self.trunk  = MLP(obs_dim, hidden_dim)
        self.policy = _ortho_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.value  = _ortho_init(nn.Linear(hidden_dim, 1),          gain=1.00)

    # ------------------------------------------------------------------ #
    #  Rollout-time inference (no grad required)                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def get_action(
        self,
        obs: "torch.Tensor",
        deterministic: bool = False,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Sample (or take the greedy) action from the policy.

        Returns
        -------
        action   : LongTensor (batch,)
        log_prob : FloatTensor (batch,)
        value    : FloatTensor (batch,)
        """
        features = self.trunk(obs)
        logits   = self.policy(features)
        v        = self.value(features).squeeze(-1)
        dist     = Categorical(logits=logits)
        action   = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), v

    # ------------------------------------------------------------------ #
    #  Update-time (grad enabled)                                          #
    # ------------------------------------------------------------------ #

    def evaluate_actions(
        self,
        obs:     "torch.Tensor",
        actions: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Re-evaluate stored (obs, action) pairs under the current policy.

        Returns
        -------
        log_prob : FloatTensor (batch,)
        value    : FloatTensor (batch,)
        entropy  : FloatTensor (batch,)
        """
        features = self.trunk(obs)
        logits   = self.policy(features)
        v        = self.value(features).squeeze(-1)
        dist     = Categorical(logits=logits)
        return dist.log_prob(actions), v, dist.entropy()

    def forward(
        self,
        obs: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Raw forward pass → (logits, value). Useful for debugging."""
        features = self.trunk(obs)
        return self.policy(features), self.value(features).squeeze(-1)


# ─────────────────────── LSTM actor-critic ───────────────────

class LSTMActorCritic(nn.Module):
    """
    Optional LSTM-based Actor-Critic for capturing temporal dependencies.
    Suitable for environments where the Markov assumption is weaker.

    Usage
    -----
    Pass `use_lstm=True` to `PPOAgent` to select this architecture instead
    of the standard MLP-based `ActorCritic`.
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
    ) -> None:
        _check_torch()
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            _ortho_init(nn.Linear(obs_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
        )
        # LSTM
        self.lstm = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = lstm_layers,
            batch_first = True,
        )
        # Heads
        self.policy = _ortho_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.value  = _ortho_init(nn.Linear(hidden_dim, 1),          gain=1.00)

        self.hidden_dim  = hidden_dim
        self.lstm_layers = lstm_layers
        self._hidden     = None   # maintained across rollout steps

    def reset_hidden(self, batch_size: int = 1, device: str = "cpu") -> None:
        self._hidden = (
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device),
        )

    @torch.no_grad()
    def get_action(
        self,
        obs: "torch.Tensor",
        deterministic: bool = False,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        if self._hidden is None:
            self.reset_hidden(batch_size=obs.shape[0], device=str(obs.device))
        x, self._hidden = self.lstm(
            self.input_proj(obs).unsqueeze(1), self._hidden
        )
        x = x.squeeze(1)
        dist   = Categorical(logits=self.policy(x))
        action = self.policy(x).argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), self.value(x).squeeze(-1)

    def evaluate_actions(
        self,
        obs:     "torch.Tensor",
        actions: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        # For batch updates, reset hidden state (treat each sample independently)
        self.reset_hidden(batch_size=obs.shape[0], device=str(obs.device))
        x, _ = self.lstm(self.input_proj(obs).unsqueeze(1), self._hidden)
        x     = x.squeeze(1)
        dist  = Categorical(logits=self.policy(x))
        return dist.log_prob(actions), self.value(x).squeeze(-1), dist.entropy()
