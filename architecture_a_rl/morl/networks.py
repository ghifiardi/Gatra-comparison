from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PreferenceConditionedActor(nn.Module):
    def __init__(self, state_dim: int, k_objectives: int, hidden: list[int], action_dim: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.k_objectives = k_objectives
        self.action_dim = action_dim
        self.net = _mlp([state_dim + k_objectives] + hidden + [action_dim])

    def forward(self, state_and_pref: torch.Tensor) -> torch.Tensor:
        logits = self.net(state_and_pref)
        return torch.softmax(logits, dim=-1)


class PreferenceConditionedVectorCritic(nn.Module):
    def __init__(self, state_dim: int, k_objectives: int, hidden: list[int]):
        super().__init__()
        self.state_dim = state_dim
        self.k_objectives = k_objectives
        self.net = _mlp([state_dim + k_objectives] + hidden + [k_objectives])

    def forward(self, state_and_pref: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(state_and_pref)
        return out
