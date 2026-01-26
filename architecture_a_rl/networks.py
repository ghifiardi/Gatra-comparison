from __future__ import annotations
import torch
import torch.nn as nn


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden: list[int], action_dim: int):
        super().__init__()
        self.net = mlp([state_dim] + hidden + [action_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: list[int]):
        super().__init__()
        self.net = mlp([state_dim] + hidden + [1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x).squeeze(-1)
        return result
