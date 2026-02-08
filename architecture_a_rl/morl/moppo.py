from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from numpy.typing import NDArray
from torch.optim import Adam

from .buffers import concat_state_pref, iter_minibatches, select_rollout
from .networks import PreferenceConditionedActor, PreferenceConditionedVectorCritic
from .objectives import ObjectiveSpec, parse_objectives
from .objectives_realdata import RealDataRewardResult, compute_reward_matrix_realdata_aware
from .preferences import normalize_weight_grid, sample_dirichlet_weights
from .types import MORLConfig, MORLRollout, MOPPOTrainConfig, PreferenceConfig


def _parse_config(morl_cfg_path: str) -> tuple[MORLConfig, list[ObjectiveSpec], dict[str, Any]]:
    with open(morl_cfg_path, "r") as f:
        cfg = cast(dict[str, Any], yaml.safe_load(f))

    morl = cast(dict[str, Any], cfg.get("morl", {}))
    train_raw = cast(dict[str, Any], morl.get("training", {}))
    pref_raw = cast(dict[str, Any], morl.get("preference_conditioning", {}))

    objectives = parse_objectives(cast(list[dict[str, object]], morl.get("objectives", [])))
    k = int(morl.get("k_objectives", len(objectives)))
    if k <= 0:
        raise ValueError("morl.k_objectives must be positive")
    if len(objectives) != k:
        raise ValueError(f"Expected {k} objectives, got {len(objectives)}")

    train_cfg = MOPPOTrainConfig(
        epochs=int(train_raw.get("epochs", 5)),
        batch_size=int(train_raw.get("batch_size", 256)),
        gamma=float(train_raw.get("gamma", 0.99)),
        gae_lambda=float(train_raw.get("gae_lambda", 0.95)),
        clip_eps=float(train_raw.get("clip_eps", 0.2)),
        entropy_coef=float(train_raw.get("entropy_coef", 0.01)),
        value_coef=float(train_raw.get("value_coef", 0.5)),
        lr=float(train_raw.get("lr", 3e-4)),
    )
    pref_cfg = PreferenceConfig(
        enabled=bool(pref_raw.get("enabled", True)),
        method=str(pref_raw.get("method", "concat_to_state")),
        dirichlet_alpha=[float(v) for v in cast(list[float], pref_raw.get("dirichlet_alpha", []))],
        sample_per=str(pref_raw.get("sample_per", "episode")),
    )
    morl_cfg = MORLConfig(
        enabled=bool(morl.get("enabled", False)),
        k_objectives=k,
        hidden=[int(v) for v in cast(list[int], train_raw.get("hidden", [256, 128, 64]))],
        train=train_cfg,
        pref=pref_cfg,
    )
    return morl_cfg, objectives, cfg


def _vector_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have the same shape")

    n, k = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros((k,), dtype=rewards.dtype, device=rewards.device)

    for t in range(n - 1, -1, -1):
        if t + 1 < n:
            next_value = values[t + 1]
        else:
            next_value = torch.zeros((k,), dtype=values.dtype, device=values.device)
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        adv[t] = last_gae

    returns = adv + values
    return adv, returns


def _build_rollout(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int_],
    actor: PreferenceConditionedActor,
    critic: PreferenceConditionedVectorCritic,
    objectives: list[ObjectiveSpec],
    pref_weights: NDArray[np.float32],
    gamma: float,
    gae_lambda: float,
    contract_dir: str | None,
    split: str,
    realdata_normalization: str,
) -> tuple[MORLRollout, RealDataRewardResult]:
    states = torch.tensor(x_train, dtype=torch.float32)
    weights_t = torch.tensor(pref_weights, dtype=torch.float32)
    inputs = concat_state_pref(states, weights_t)

    with torch.no_grad():
        probs = actor(inputs)
        dist = cast(Any, torch.distributions.Categorical)(probs=probs)
        actions = dist.sample()
        old_logp = dist.log_prob(actions)
        values = critic(inputs)

    actions_np = actions.cpu().numpy().astype(np.int_)
    reward_result = compute_reward_matrix_realdata_aware(
        y_true=y_train.astype(np.int_),
        actions=actions_np,
        objectives=objectives,
        contract_dir=contract_dir,
        split=split,
        normalization=realdata_normalization,
    )
    rewards_vec_np = reward_result.reward_matrix
    rewards_vec = torch.tensor(rewards_vec_np, dtype=torch.float32)

    dones = torch.ones((x_train.shape[0],), dtype=torch.float32)
    adv_vec, returns_vec = _vector_gae(
        rewards_vec, values, dones, gamma=gamma, gae_lambda=gae_lambda
    )

    adv_scalar = torch.sum(adv_vec * weights_t, dim=1)
    adv_scalar = (adv_scalar - adv_scalar.mean()) / (adv_scalar.std() + 1e-8)

    return MORLRollout(
        inputs=inputs,
        weights=weights_t,
        actions=actions,
        old_logp=old_logp,
        rewards_vec=rewards_vec,
        advantages_vec=adv_vec,
        advantages_scalar=adv_scalar,
        returns_vec=returns_vec,
    ), reward_result


def _update_batch(
    actor: PreferenceConditionedActor,
    critic: PreferenceConditionedVectorCritic,
    optim: Adam,
    batch: MORLRollout,
    cfg: MOPPOTrainConfig,
) -> dict[str, float]:
    probs = actor(batch.inputs)
    dist = cast(Any, torch.distributions.Categorical)(probs=probs)
    logp = dist.log_prob(batch.actions)

    ratio = torch.exp(logp - batch.old_logp)
    unclipped = ratio * batch.advantages_scalar
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * batch.advantages_scalar
    policy_loss = -torch.min(unclipped, clipped).mean()

    values = critic(batch.inputs)
    value_loss = F.mse_loss(values, batch.returns_vec)

    entropy = dist.entropy().mean()
    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(actor.parameters()) + list(critic.parameters()), max_norm=0.5
    )
    optim.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
    }


def train_moppo_from_arrays(
    morl_cfg_path: str,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int_],
    out_dir: str,
    *,
    seed: int,
    x_val: NDArray[np.float32] | None = None,
    y_val: NDArray[np.int_] | None = None,
    contract_dir: str | None = None,
) -> str:
    del x_val, y_val

    os.makedirs(out_dir, exist_ok=True)
    cfg, objectives, raw_cfg = _parse_config(morl_cfg_path)
    if not cfg.enabled:
        raise ValueError("MORL config is disabled; set morl.enabled=true")

    if x_train.shape[1] != 128:
        raise ValueError(f"Expected v128 inputs, got shape {x_train.shape}")

    if len(cfg.pref.dirichlet_alpha) != cfg.k_objectives:
        raise ValueError("preference_conditioning.dirichlet_alpha length must equal k_objectives")

    torch.manual_seed(seed)
    np.random.seed(seed)

    actor = PreferenceConditionedActor(
        state_dim=128,
        k_objectives=cfg.k_objectives,
        hidden=cfg.hidden,
        action_dim=2,
    )
    critic = PreferenceConditionedVectorCritic(
        state_dim=128,
        k_objectives=cfg.k_objectives,
        hidden=cfg.hidden,
    )
    optim = Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.train.lr)

    logs: list[dict[str, float]] = []
    n = x_train.shape[0]
    realdata_cfg = cast(dict[str, Any], raw_cfg.get("morl", {})).get("realdata_objectives", {})
    realdata_norm = "minmax"
    if isinstance(realdata_cfg, dict):
        realdata_norm = str(realdata_cfg.get("normalization", "minmax"))
    objective_source = "fallback_synthetic"
    objective_stats: dict[str, float] = {}

    for epoch in range(cfg.train.epochs):
        weights = sample_dirichlet_weights(cfg.pref.dirichlet_alpha, n_samples=n, seed=seed + epoch)
        rollout, reward_info = _build_rollout(
            x_train=x_train,
            y_train=y_train,
            actor=actor,
            critic=critic,
            objectives=objectives,
            pref_weights=weights,
            gamma=cfg.train.gamma,
            gae_lambda=cfg.train.gae_lambda,
            contract_dir=contract_dir,
            split="train",
            realdata_normalization=realdata_norm,
        )
        objective_source = reward_info.source
        objective_stats = reward_info.stats

        rng = np.random.default_rng(seed + 1000 + epoch)
        losses: list[float] = []
        plosses: list[float] = []
        vlosses: list[float] = []
        entropies: list[float] = []

        for idx in iter_minibatches(n, cfg.train.batch_size, rng):
            mb = select_rollout(rollout, idx)
            stats = _update_batch(actor, critic, optim, mb, cfg.train)
            losses.append(stats["loss"])
            plosses.append(stats["policy_loss"])
            vlosses.append(stats["value_loss"])
            entropies.append(stats["entropy"])

        logs.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "policy_loss": float(np.mean(plosses)) if plosses else 0.0,
                "value_loss": float(np.mean(vlosses)) if vlosses else 0.0,
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                "n_batches": float(len(losses)),
            }
        )

    actor_path = os.path.join(out_dir, "actor.pt")
    critic_path = os.path.join(out_dir, "critic.pt")
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)

    meta = {
        "model_name": "moppo",
        "k_objectives": cfg.k_objectives,
        "objectives": [asdict(o) for o in objectives],
        "seed": seed,
        "preference_conditioning_enabled": cfg.pref.enabled,
        "preference_method": cfg.pref.method,
        "sample_per": cfg.pref.sample_per,
        "state_dim": 128,
        "action_dim": 2,
        "hidden": cfg.hidden,
        "train_samples": int(x_train.shape[0]),
        "objective_source": objective_source,
        "realdata_objective_stats": objective_stats,
        "config": raw_cfg,
    }
    meta_path = os.path.join(out_dir, "morl_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump({"epochs": logs}, f, indent=2)

    return actor_path


def load_weight_grid_from_config(morl_cfg_path: str) -> NDArray[np.float32]:
    with open(morl_cfg_path, "r") as f:
        cfg = cast(dict[str, Any], yaml.safe_load(f))
    morl = cast(dict[str, Any], cfg.get("morl", {}))
    k = int(morl.get("k_objectives", 3))
    eval_cfg = cast(dict[str, Any], morl.get("eval", {}))
    sweep_cfg = cast(dict[str, Any], eval_cfg.get("weight_sweep", {}))
    grid = cast(list[list[float]], sweep_cfg.get("grid", []))
    return normalize_weight_grid(grid, k)
