"""Head-to-head evaluation comparing RL (PPO) and Isolation Forest models.

This module runs both models on the same test set and computes classification
metrics for fair comparison. Key features:

- **Separate latency measurements**: Each model's inference time is measured
  independently to account for different feature extraction costs.
- **Tuned thresholds**: Uses thresholds tuned on validation set (if available)
  from saved checkpoints for fair comparison.
- **Consistent scoring**: Both models produce threat scores in [0, 1] range.
"""
from __future__ import annotations
import os
import json
import time
import numpy as np
import yaml
import joblib
import torch
from data.loaders import load_data
from data.splits import time_split
from data.features import extract_features_v7, extract_features_v128, HistoryContext
from architecture_a_rl.networks import Actor
from architecture_b_iforest.model import IForestModel
from .metrics import classification_metrics

ACTIONS = ["escalate", "contain", "monitor", "dismiss"]


def run_head_to_head(eval_cfg: str, data_cfg: str, iforest_cfg: str, ppo_cfg: str) -> str:
    """Run head-to-head evaluation of RL vs Isolation Forest.

    Args:
        eval_cfg: Path to evaluation config YAML.
        data_cfg: Path to data config YAML.
        iforest_cfg: Path to Isolation Forest config YAML.
        ppo_cfg: Path to PPO config YAML.

    Returns:
        Path to the generated report JSON file.
    """
    with open(eval_cfg, "r") as f:
        e_cfg = yaml.safe_load(f)
    with open(iforest_cfg, "r") as f:
        i_cfg = yaml.safe_load(f)
    with open(ppo_cfg, "r") as f:
        p_cfg = yaml.safe_load(f)

    out_dir = e_cfg["evaluation"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    loaded = load_data(data_cfg)
    splits = time_split(loaded.events, loaded.labels, data_cfg)
    test_events, test_labels = splits.test
    label_map = {lb.event_id: lb for lb in test_labels}

    # Load IF bundle
    if_bundle_path = os.path.join(i_cfg["io"]["output_dir"], "iforest_bundle.joblib")
    if_bundle = joblib.load(if_bundle_path)
    prep = if_bundle["preprocessor"]
    ifm: IForestModel = if_bundle["model"]

    # Use tuned threshold from bundle if available, otherwise fall back to config
    if_threshold = ifm.threshold

    # Load PPO
    ppo_path = os.path.join(p_cfg["io"]["output_dir"], "ppo_policy.pt")
    ckpt = torch.load(ppo_path, map_location="cpu")
    actor = Actor(
        state_dim=p_cfg["rl"]["state_dim"],
        hidden=p_cfg["networks"]["hidden_sizes"],
        action_dim=4,
    )
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    # Use tuned threshold from checkpoint if available, otherwise fall back to config
    rl_threshold = ckpt.get("threshold", e_cfg["head_to_head"]["rl_threshold"])

    # Filter valid test samples
    valid_events = []
    valid_labels_binary = []
    for e in test_events:
        lb = label_map.get(e.event_id)
        if lb is None or lb.label == "unknown":
            continue
        valid_events.append(e)
        valid_labels_binary.append(1 if lb.label == "threat" else 0)

    y_true = np.array(valid_labels_binary, dtype=int)
    n_samples = len(valid_events)

    # =========================================================================
    # Isolation Forest inference with separate latency measurement
    # =========================================================================
    y_if_score_list: list[float] = []
    t0_if = time.perf_counter()
    for e in valid_events:
        x7 = extract_features_v7(e)[None, :]
        x7p = prep.transform(x7)
        s_if = float(ifm.score(x7p)[0])
        y_if_score_list.append(s_if)
    t1_if = time.perf_counter()
    if_latency_ms = (t1_if - t0_if) * 1000.0 / max(1, n_samples)

    # =========================================================================
    # RL (PPO) inference with separate latency measurement
    # =========================================================================
    y_rl_score_list: list[float] = []
    t0_rl = time.perf_counter()
    with torch.no_grad():
        for e in valid_events:
            s128 = extract_features_v128(e, HistoryContext(now=e.ts))
            st = torch.tensor(s128, dtype=torch.float32).unsqueeze(0)
            probs = actor(st).squeeze(0).numpy()
            # Threat score proxy: sum prob of (escalate, contain)
            y_rl_score_list.append(float(probs[0] + probs[1]))
    t1_rl = time.perf_counter()
    rl_latency_ms = (t1_rl - t0_rl) * 1000.0 / max(1, n_samples)

    y_if_score = np.array(y_if_score_list, dtype=float)
    y_rl_score = np.array(y_rl_score_list, dtype=float)

    # Compute classification metrics
    m_if = classification_metrics(y_true, y_if_score, threshold=if_threshold)
    m_rl = classification_metrics(y_true, y_rl_score, threshold=rl_threshold)

    # Build report with threshold info
    if_threshold_tuned = if_bundle.get("meta", {}).get("threshold_tuned", False)
    rl_threshold_tuned = ckpt.get("threshold_tuned", False)

    iforest_report: dict[str, float | str] = {
        **m_if,
        "threshold": if_threshold,
        "threshold_source": "tuned" if if_threshold_tuned else "config",
    }
    rl_report: dict[str, float | str] = {
        **m_rl,
        "threshold": rl_threshold,
        "threshold_source": "tuned" if rl_threshold_tuned else "config",
    }

    report = {
        "n_test": n_samples,
        "iforest": iforest_report,
        "rl": rl_report,
        "latency": {
            "iforest_ms_per_event": float(if_latency_ms),
            "rl_ms_per_event": float(rl_latency_ms),
            "note": "Measured separately including feature extraction for each model",
        },
    }
    path = os.path.join(out_dir, "head_to_head_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
