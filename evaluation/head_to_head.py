from __future__ import annotations

import os
import json
import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray
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


def _load_contract_arrays(
    contract_dir: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int_]]:
    x7 = np.load(os.path.join(contract_dir, "features_v7.npy"))
    x128 = np.load(os.path.join(contract_dir, "features_v128.npy"))
    y_true = np.load(os.path.join(contract_dir, "y_true.npy"))
    return x7.astype(np.float32), x128.astype(np.float32), y_true.astype(np.int_)


def run_head_to_head(
    eval_cfg: str,
    data_cfg: str,
    iforest_cfg: str,
    ppo_cfg: str,
    contract_dir: Optional[str] = None,
) -> str:
    e_cfg = yaml.safe_load(open(eval_cfg, "r"))
    i_cfg = yaml.safe_load(open(iforest_cfg, "r"))
    p_cfg = yaml.safe_load(open(ppo_cfg, "r"))

    out_dir = e_cfg["evaluation"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Load IF bundle
    if_bundle_path = os.path.join(i_cfg["io"]["output_dir"], "iforest_bundle.joblib")
    if_bundle = joblib.load(if_bundle_path)
    prep = if_bundle["preprocessor"]
    ifm: IForestModel = if_bundle["model"]

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

    y_rl_score_list: list[float]
    if contract_dir:
        x7, x128, y_true_arr = _load_contract_arrays(contract_dir)

        t0 = time.time()
        x7p = prep.transform(x7)
        y_if_score_arr = np.asarray(ifm.score(x7p), dtype=float)

        y_rl_score_list = []
        with torch.no_grad():
            for row in x128:
                st = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
                probs = actor(st).squeeze(0).numpy()
                y_rl_score_list.append(float(probs[0] + probs[1]))
        t1 = time.time()
        infer_ms = (t1 - t0) * 1000.0 / max(1, len(y_true_arr))

        y_rl_score_arr = np.asarray(y_rl_score_list, dtype=float)
    else:
        loaded = load_data(data_cfg)
        splits = time_split(loaded.events, loaded.labels, data_cfg)
        test_events, test_labels = splits.test
        label_map = {lb.event_id: lb for lb in test_labels}

        y_true_list: list[int] = []
        y_if_score_list: list[float] = []
        y_rl_score_list = []

        t0 = time.time()
        for e in test_events:
            lb = label_map.get(e.event_id)
            if lb is None or lb.label == "unknown":
                continue
            y_true_list.append(1 if lb.label == "threat" else 0)

            x7 = extract_features_v7(e)[None, :]
            x7p = prep.transform(x7)
            s_if = float(ifm.score(x7p)[0])
            y_if_score_list.append(s_if)

            s128 = extract_features_v128(e, HistoryContext(now=e.ts))
            st = torch.tensor(s128, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = actor(st).squeeze(0).numpy()
            y_rl_score_list.append(float(probs[0] + probs[1]))

        t1 = time.time()
        infer_ms = (t1 - t0) * 1000.0 / max(1, len(y_true_list))

        y_true_arr = np.array(y_true_list, dtype=int)
        y_if_score_arr = np.array(y_if_score_list, dtype=float)
        y_rl_score_arr = np.array(y_rl_score_list, dtype=float)

    m_if = classification_metrics(
        y_true_arr, y_if_score_arr, threshold=float(i_cfg["scoring"]["threshold"])
    )
    m_rl = classification_metrics(y_true_arr, y_rl_score_arr, threshold=0.5)

    report = {
        "n_test": int(len(y_true_arr)),
        "iforest": m_if,
        "rl": m_rl,
        "infer_latency_ms_per_event": float(infer_ms),
        "contract_dir": contract_dir,
    }
    path = os.path.join(out_dir, "head_to_head_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
