from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from .networks import Actor, Critic


@dataclass
class PPOConfig:
    lr: float
    clip_ratio: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float


def ppo_update(
    actor: Actor,
    critic: Critic,
    optim: Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: PPOConfig,
) -> dict[str, float]:
    states, actions, old_logp, returns = batch
    probs = actor(states)
    dist = torch.distributions.Categorical(probs=probs)
    logp = dist.log_prob(actions)  # type: ignore[no-untyped-call]

    ratio = torch.exp(logp - old_logp)
    adv = returns - critic(states).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    clip_adv = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv
    policy_loss = -(torch.min(ratio * adv, clip_adv)).mean()

    values = critic(states)
    value_loss = F.mse_loss(values, returns)

    entropy = dist.entropy().mean()  # type: ignore[no-untyped-call]
    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm
    )
    optim.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
    }
