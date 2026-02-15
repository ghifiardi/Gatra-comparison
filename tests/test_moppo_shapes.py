from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from architecture_a_rl.morl.moppo import train_moppo_from_arrays
from architecture_a_rl.morl.networks import (
    PreferenceConditionedActor,
    PreferenceConditionedVectorCritic,
)


def _write_morl_cfg(path: Path) -> None:
    path.write_text(
        """
morl:
  enabled: true
  k_objectives: 3
  objectives:
    - name: detect
      type: classification
      tp: 1.0
      fn: -1.0
      fp: 0.0
      tn: 0.0
    - name: fp_cost
      type: fp_penalty
      fp_penalty: -0.2
    - name: analyst_cost
      type: per_alert_cost
      per_alert_penalty: -0.05
  preference_conditioning:
    enabled: true
    method: concat_to_state
    dirichlet_alpha: [0.7, 0.7, 0.7]
    sample_per: episode
  training:
    algorithm: moppo
    epochs: 1
    batch_size: 16
    gamma: 0.99
    gae_lambda: 0.95
    clip_eps: 0.2
    entropy_coef: 0.01
    value_coef: 0.5
    lr: 0.0003
    hidden: [32, 16]
  eval:
    weight_sweep:
      enabled: true
      grid:
        - [0.7, 0.2, 0.1]
    pareto:
      enabled: true
      primary_metrics: ["pr_auc", "f1", "alerts_per_1k"]
    hypervolume:
      enabled: true
      reference: [0.0, 0.0, 1000.0]
""".strip()
    )


def test_moppo_smoke_shapes_and_artifacts(tmp_path: Path) -> None:
    cfg = tmp_path / "morl.yaml"
    _write_morl_cfg(cfg)

    n = 64
    x = np.random.default_rng(42).normal(size=(n, 128)).astype(np.float32)
    y = np.random.default_rng(7).integers(0, 2, size=(n,), endpoint=False).astype(np.int_)

    out_dir = tmp_path / "morl_model"
    actor_path = train_moppo_from_arrays(
        str(cfg),
        x,
        y,
        str(out_dir),
        seed=42,
    )

    assert Path(actor_path).exists()
    assert (out_dir / "critic.pt").exists()
    assert (out_dir / "morl_meta.json").exists()
    assert (out_dir / "train_log.json").exists()

    with open(out_dir / "morl_meta.json", "r") as f:
        meta = json.load(f)
    assert meta["k_objectives"] == 3

    actor = PreferenceConditionedActor(state_dim=128, k_objectives=3, hidden=[32, 16], action_dim=2)
    critic = PreferenceConditionedVectorCritic(state_dim=128, k_objectives=3, hidden=[32, 16])

    xw = torch.zeros((4, 131), dtype=torch.float32)
    probs = actor(xw)
    values = critic(xw)

    assert probs.shape == (4, 2)
    assert values.shape == (4, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)
