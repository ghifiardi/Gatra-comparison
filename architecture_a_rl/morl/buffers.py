from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from .types import MORLRollout


def concat_state_pref(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if states.ndim != 2:
        raise ValueError("states must be rank-2")
    if weights.ndim != 2:
        raise ValueError("weights must be rank-2")
    if states.shape[0] != weights.shape[0]:
        raise ValueError("states and weights must have identical first dimension")
    return torch.cat([states, weights], dim=1)


def iter_minibatches(n: int, batch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    idx = np.arange(n)
    rng.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i : i + batch_size]


def select_rollout(rollout: MORLRollout, idx: np.ndarray) -> MORLRollout:
    t = torch.from_numpy(idx).long()
    return MORLRollout(
        inputs=rollout.inputs[t],
        weights=rollout.weights[t],
        actions=rollout.actions[t],
        old_logp=rollout.old_logp[t],
        rewards_vec=rollout.rewards_vec[t],
        advantages_vec=rollout.advantages_vec[t],
        advantages_scalar=rollout.advantages_scalar[t],
        returns_vec=rollout.returns_vec[t],
    )
