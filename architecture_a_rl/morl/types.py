from __future__ import annotations

from dataclasses import dataclass

import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class MOPPOTrainConfig:
    epochs: int
    batch_size: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    entropy_coef: float
    value_coef: float
    lr: float


@dataclass(frozen=True)
class PreferenceConfig:
    enabled: bool
    method: str
    dirichlet_alpha: list[float]
    sample_per: str


@dataclass(frozen=True)
class MORLConfig:
    enabled: bool
    k_objectives: int
    hidden: list[int]
    train: MOPPOTrainConfig
    pref: PreferenceConfig


@dataclass(frozen=True)
class MORLRollout:
    inputs: torch.Tensor
    weights: torch.Tensor
    actions: torch.Tensor
    old_logp: torch.Tensor
    rewards_vec: torch.Tensor
    advantages_vec: torch.Tensor
    advantages_scalar: torch.Tensor
    returns_vec: torch.Tensor


Float32Array = NDArray["float32"]
