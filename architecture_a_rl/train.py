"""PPO training module for threat response policy.

DESIGN DECISION: Contextual Bandit vs. Full RL
==============================================

This implementation uses PPO in a **contextual bandit** setting rather than
full reinforcement learning. Key differences:

1. **No Sequential Dependencies**: Each security event is treated independently.
   The action taken on one event doesn't affect future events or states.
   This is appropriate for SOC alert triage where alerts arrive independently.

2. **Immediate Rewards**: Rewards are computed immediately based on the action
   taken and the ground truth label. There's no delayed reward or credit
   assignment problem.

3. **Offline Data Collection**: Actions are sampled once from the initial policy
   before training begins. The policy is then optimized on this fixed dataset
   (offline RL / batch RL). This differs from online PPO where the policy
   would be updated between environment interactions.

4. **Why PPO for Bandits?**: PPO's clipped objective and entropy regularization
   still provide benefits:
   - Stable policy updates (prevents drastic changes)
   - Entropy bonus encourages exploration
   - Value function helps reduce variance in advantage estimation

For a full RL formulation, you would need:
- State transitions (e.g., SOC workload, analyst fatigue)
- Delayed rewards (e.g., incident resolution time)
- Episode structure with terminal states

This bandit formulation is a deliberate simplification that's appropriate
when the primary goal is per-event classification/triage.
"""

from __future__ import annotations
import os
import json
from typing import Any, cast
import numpy as np
from numpy.typing import NDArray
import yaml
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.optim import Adam
from data.loaders import load_data
from data.splits import time_split
from data.features import extract_features_v128, HistoryContext
from data.schemas import RawEvent, Label, LabelType
from .networks import Actor, Critic
from .env_bandit import compute_reward, RewardConfig, ACTIONS
from .ppo import ppo_update, PPOConfig

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.int_]


def _prepare_dataset(
    events: list[RawEvent],
    labels: list[Label],
    actor: Actor,
    rcfg: RewardConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare dataset tensors from events and labels."""
    label_map = {lb.event_id: lb for lb in labels}

    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    old_logps: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    for e in events:
        lb = label_map.get(e.event_id)
        if lb is None:
            continue
        s = extract_features_v128(e, HistoryContext(now=e.ts))
        s_t = torch.tensor(s, dtype=torch.float32)

        with torch.no_grad():
            probs = actor(s_t.unsqueeze(0)).squeeze(0)
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()  # type: ignore[no-untyped-call]
            logp = dist.log_prob(a)  # type: ignore[no-untyped-call]

        action_name = ACTIONS[int(a.item())]
        r = compute_reward(lb.label, float(lb.severity), action_name, rcfg)

        states.append(s_t)
        actions.append(a)
        old_logps.append(logp)
        rewards.append(torch.tensor(r, dtype=torch.float32))

    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(old_logps),
        torch.stack(rewards),
    )


def _get_threat_scores_and_labels(
    events: list[RawEvent],
    labels: list[Label],
    actor: Actor,
) -> tuple[IntArray, FloatArray]:
    """Extract threat scores and ground truth labels for threshold tuning.

    The threat score is computed as: P(escalate) + P(contain), i.e., the
    probability that the policy recommends a response action for a threat.
    """
    label_map = {lb.event_id: lb for lb in labels}

    y_true: list[int] = []
    y_scores: list[float] = []

    actor.eval()
    with torch.no_grad():
        for e in events:
            lb = label_map.get(e.event_id)
            if lb is None or lb.label == "unknown":
                continue

            y_true.append(1 if lb.label == "threat" else 0)

            s = extract_features_v128(e, HistoryContext(now=e.ts))
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            probs = actor(s_t).squeeze(0).numpy()

            # Threat score = P(escalate) + P(contain)
            threat_score = float(probs[0] + probs[1])
            y_scores.append(threat_score)

    actor.train()
    return np.array(y_true, dtype=int), np.array(y_scores, dtype=float)


def _prepare_dataset_from_arrays(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int_],
    actor: Actor,
    rcfg: RewardConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare PPO tensors from precomputed feature arrays."""
    states_t = torch.tensor(x_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.int64)

    with torch.no_grad():
        probs = actor(states_t)
        dist = cast(Any, torch.distributions.Categorical)(probs=probs)
        actions = dist.sample()
        old_logps = dist.log_prob(actions)

    rewards: list[float] = []
    for i in range(states_t.shape[0]):
        label: LabelType = "threat" if int(labels[i].item()) == 1 else "benign"
        severity = 1.0 if label == "threat" else 0.0
        action_name = ACTIONS[int(actions[i].item())]
        rewards.append(float(compute_reward(label, severity, action_name, rcfg)))

    returns_t = torch.tensor(rewards, dtype=torch.float32)
    return states_t, actions, old_logps, returns_t


def _get_threat_scores_from_arrays(
    x: NDArray[np.float32],
    y: NDArray[np.int_],
    actor: Actor,
) -> tuple[IntArray, FloatArray]:
    """Compute threat scores from feature arrays."""
    y_true = y.astype(int)
    scores: list[float] = []
    with torch.no_grad():
        for row in x:
            st = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            probs = actor(st).squeeze(0).numpy()
            scores.append(float(probs[0] + probs[1]))
    return y_true, np.array(scores, dtype=float)


def _tune_threshold_on_validation(
    y_true: IntArray,
    y_scores: FloatArray,
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Find optimal threshold that maximizes F1 score on validation set.

    This ensures fair comparison with the Isolation Forest baseline by using
    the same threshold tuning approach for both models.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics: dict[str, float] = {}

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            p, r, _, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            best_metrics = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "threshold": thresh,
            }

    return best_threshold, best_metrics


def _evaluate_on_split(
    events: list[RawEvent],
    labels: list[Label],
    actor: Actor,
    rcfg: RewardConfig,
) -> dict[str, float]:
    """Evaluate actor on a data split and return metrics."""
    label_map = {lb.event_id: lb for lb in labels}

    total_reward = 0.0
    correct_actions = 0
    total_samples = 0

    actor.eval()
    with torch.no_grad():
        for e in events:
            lb = label_map.get(e.event_id)
            if lb is None:
                continue

            s = extract_features_v128(e, HistoryContext(now=e.ts))
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            probs = actor(s_t).squeeze(0)

            # Use argmax for evaluation (deterministic)
            a = int(torch.argmax(probs).item())
            action_name = ACTIONS[a]
            r = compute_reward(lb.label, float(lb.severity), action_name, rcfg)

            total_reward += r
            total_samples += 1

            # Check if action is "correct"
            is_threat = lb.label == "threat"
            is_response_action = action_name in ("escalate", "contain")
            if (is_threat and is_response_action) or (not is_threat and not is_response_action):
                correct_actions += 1

    actor.train()

    return {
        "mean_reward": total_reward / max(1, total_samples),
        "accuracy": correct_actions / max(1, total_samples),
        "n_samples": total_samples,
    }


def train_ppo(ppo_cfg_path: str, data_cfg_path: str) -> str:
    with open(ppo_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["io"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    seed = int(cfg["rl"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    loaded = load_data(data_cfg_path)
    splits = time_split(loaded.events, loaded.labels, data_cfg_path)
    train_events, train_labels = splits.train
    val_events, val_labels = splits.val

    hidden = list(cfg["networks"]["hidden_sizes"])
    actor = Actor(
        state_dim=cfg["rl"]["state_dim"], hidden=hidden, action_dim=cfg["rl"]["action_dim"]
    )
    critic = Critic(state_dim=cfg["rl"]["state_dim"], hidden=hidden)
    optim = Adam(list(actor.parameters()) + list(critic.parameters()), lr=float(cfg["train"]["lr"]))

    rcfg = RewardConfig(
        tp_base=float(cfg["reward"]["tp_base"]),
        fp_base=float(cfg["reward"]["fp_base"]),
        fn_base=float(cfg["reward"]["fn_base"]),
        efficiency_bonus=float(cfg["reward"]["efficiency_bonus"]),
        action_cost={k: float(v) for k, v in cfg["reward"]["action_cost"].items()},
    )
    pcfg = PPOConfig(
        lr=float(cfg["train"]["lr"]),
        clip_ratio=float(cfg["train"]["clip_ratio"]),
        entropy_coef=float(cfg["train"]["entropy_coef"]),
        value_coef=float(cfg["train"]["value_coef"]),
        max_grad_norm=float(cfg["train"]["max_grad_norm"]),
    )

    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    eval_every = int(cfg.get("validation", {}).get("eval_every_epoch", 1))
    early_stop_patience = int(cfg.get("validation", {}).get("early_stop_patience", epochs))

    # Prepare training dataset
    states_t, actions_t, old_logps_t, returns_t = _prepare_dataset(
        train_events, train_labels, actor, rcfg
    )

    n = states_t.shape[0]
    idx = np.arange(n)

    logs: list[dict[str, float]] = []
    best_val_reward = float("-inf")
    patience_counter = 0
    best_actor_state = actor.state_dict()
    best_critic_state = critic.state_dict()

    for ep in range(epochs):
        # Training
        np.random.shuffle(idx)
        epoch_losses: list[float] = []
        epoch_policy_losses: list[float] = []
        epoch_value_losses: list[float] = []
        epoch_entropies: list[float] = []

        for i in range(0, n, batch_size):
            j = idx[i : i + batch_size]
            batch = (states_t[j], actions_t[j], old_logps_t[j], returns_t[j])
            m = ppo_update(actor, critic, optim, batch, pcfg)

            # Log all batches
            epoch_losses.append(m["loss"])
            epoch_policy_losses.append(m["policy_loss"])
            epoch_value_losses.append(m["value_loss"])
            epoch_entropies.append(m["entropy"])

        # Aggregate epoch metrics (average over all batches)
        epoch_log: dict[str, float] = {
            "epoch": float(ep),
            "loss": float(np.mean(epoch_losses)),
            "policy_loss": float(np.mean(epoch_policy_losses)),
            "value_loss": float(np.mean(epoch_value_losses)),
            "entropy": float(np.mean(epoch_entropies)),
            "n_batches": float(len(epoch_losses)),
        }

        # Validation evaluation
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            val_metrics = _evaluate_on_split(val_events, val_labels, actor, rcfg)
            epoch_log["val_mean_reward"] = val_metrics["mean_reward"]
            epoch_log["val_accuracy"] = val_metrics["accuracy"]

            # Early stopping check
            if val_metrics["mean_reward"] > best_val_reward:
                best_val_reward = val_metrics["mean_reward"]
                patience_counter = 0
                best_actor_state = actor.state_dict()
                best_critic_state = critic.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                epoch_log["early_stopped"] = 1.0
                logs.append(epoch_log)
                break

        logs.append(epoch_log)

    # Restore best model
    actor.load_state_dict(best_actor_state)
    critic.load_state_dict(best_critic_state)

    # Tune threshold on validation set if enabled
    tune_threshold = cfg.get("validation", {}).get("tune_threshold", False)
    if tune_threshold:
        y_true_val, y_scores_val = _get_threat_scores_and_labels(val_events, val_labels, actor)
        best_threshold, tuning_metrics = _tune_threshold_on_validation(y_true_val, y_scores_val)
    else:
        best_threshold = 0.5  # Default threshold
        tuning_metrics = {}

    ckpt = os.path.join(out_dir, "ppo_policy.pt")
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "cfg": cfg,
            "threshold": best_threshold,
            "threshold_tuned": tune_threshold,
        },
        ckpt,
    )

    # Add threshold info to final log
    final_log: dict[str, float | bool | dict[str, float]] = {
        "final_threshold": best_threshold,
        "threshold_tuned": tune_threshold,
    }
    if tuning_metrics:
        final_log["tuning_metrics"] = tuning_metrics

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump({"epochs": logs, "final": final_log}, f, indent=2)

    return ckpt


def train_ppo_from_arrays(
    ppo_cfg_path: str,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int_],
    out_dir: str,
    x_val: NDArray[np.float32] | None = None,
    y_val: NDArray[np.int_] | None = None,
) -> str:
    """Train PPO policy from precomputed feature arrays."""
    with open(ppo_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(out_dir, exist_ok=True)

    seed = int(cfg["rl"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden = list(cfg["networks"]["hidden_sizes"])
    actor = Actor(
        state_dim=cfg["rl"]["state_dim"], hidden=hidden, action_dim=cfg["rl"]["action_dim"]
    )
    critic = Critic(state_dim=cfg["rl"]["state_dim"], hidden=hidden)
    optim = Adam(list(actor.parameters()) + list(critic.parameters()), lr=float(cfg["train"]["lr"]))

    rcfg = RewardConfig(
        tp_base=float(cfg["reward"]["tp_base"]),
        fp_base=float(cfg["reward"]["fp_base"]),
        fn_base=float(cfg["reward"]["fn_base"]),
        efficiency_bonus=float(cfg["reward"]["efficiency_bonus"]),
        action_cost={k: float(v) for k, v in cfg["reward"]["action_cost"].items()},
    )
    pcfg = PPOConfig(
        lr=float(cfg["train"]["lr"]),
        clip_ratio=float(cfg["train"]["clip_ratio"]),
        entropy_coef=float(cfg["train"]["entropy_coef"]),
        value_coef=float(cfg["train"]["value_coef"]),
        max_grad_norm=float(cfg["train"]["max_grad_norm"]),
    )

    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])

    states_t, actions_t, old_logps_t, returns_t = _prepare_dataset_from_arrays(
        x_train, y_train, actor, rcfg
    )
    n = states_t.shape[0]
    idx = np.arange(n)

    logs: list[dict[str, float]] = []
    for ep in range(epochs):
        np.random.shuffle(idx)
        epoch_losses: list[float] = []
        epoch_policy_losses: list[float] = []
        epoch_value_losses: list[float] = []
        epoch_entropies: list[float] = []

        for i in range(0, n, batch_size):
            j = idx[i : i + batch_size]
            batch = (states_t[j], actions_t[j], old_logps_t[j], returns_t[j])
            m = ppo_update(actor, critic, optim, batch, pcfg)
            epoch_losses.append(m["loss"])
            epoch_policy_losses.append(m["policy_loss"])
            epoch_value_losses.append(m["value_loss"])
            epoch_entropies.append(m["entropy"])

        epoch_log: dict[str, float] = {
            "epoch": float(ep),
            "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "policy_loss": float(np.mean(epoch_policy_losses)) if epoch_policy_losses else 0.0,
            "value_loss": float(np.mean(epoch_value_losses)) if epoch_value_losses else 0.0,
            "entropy": float(np.mean(epoch_entropies)) if epoch_entropies else 0.0,
            "n_batches": float(len(epoch_losses)),
        }
        logs.append(epoch_log)

    tune_threshold = cfg.get("validation", {}).get("tune_threshold", False)
    if tune_threshold and x_val is not None and y_val is not None:
        y_true_val, y_scores_val = _get_threat_scores_from_arrays(x_val, y_val, actor)
        best_threshold, tuning_metrics = _tune_threshold_on_validation(y_true_val, y_scores_val)
    else:
        best_threshold = 0.5
        tuning_metrics = {}

    actor_path = os.path.join(out_dir, "actor.pt")
    critic_path = os.path.join(out_dir, "critic.pt")
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)

    meta = {
        "model_name": "ppo",
        "threshold": best_threshold,
        "threshold_tuned": bool(tune_threshold and x_val is not None and y_val is not None),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]) if x_val is not None else 0,
    }
    if tuning_metrics:
        meta["tuning_metrics"] = tuning_metrics

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump({"epochs": logs}, f, indent=2)

    return actor_path
