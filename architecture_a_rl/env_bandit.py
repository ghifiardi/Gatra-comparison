from __future__ import annotations
from dataclasses import dataclass
from data.schemas import LabelType

ACTIONS = ["escalate", "contain", "monitor", "dismiss"]

@dataclass
class RewardConfig:
    tp_base: float
    fp_base: float
    fn_base: float
    efficiency_bonus: float
    action_cost: dict[str, float]

def compute_reward(
    label: LabelType,
    severity: float,
    action: str,
    cfg: RewardConfig,
) -> float:
    # Simple bandit reward shaping
    cost = cfg.action_cost.get(action, 1.0)

    if label == "threat":
        # Good if we escalate/contain, bad if dismiss/monitor
        if action in ("escalate", "contain"):
            return cfg.tp_base * severity
        else:
            return cfg.fn_base * severity
    elif label == "benign":
        # Good if we dismiss/monitor, bad if escalate/contain
        if action in ("dismiss", "monitor"):
            return cfg.efficiency_bonus
        else:
            return cfg.fp_base * cost
    else:
        # unknown labels: small penalty for heavy actions
        return -0.1 * cost
