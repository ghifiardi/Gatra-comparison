from __future__ import annotations

import json
import os
import random
import subprocess
import time
import shutil
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import torch
import typer
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from architecture_a_rl.networks import Actor
from architecture_a_rl.train import train_ppo_from_arrays
from architecture_b_iforest.model import IForestModel
from architecture_b_iforest.preprocess import Preprocessor
from architecture_b_iforest.train import train_iforest_from_arrays
from data.contract_export import export_frozen_contract_to_dir
from evaluation.metrics import classification_metrics
from runs.reporting import (
    build_run_manifest,
    dump_yaml,
    file_sha256,
    load_yaml,
    render_summary_md,
    utc_run_id,
    write_run_manifest,
)


app = typer.Typer()


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _snapshot_configs(
    data_config: str,
    iforest_config: str,
    ppo_config: str,
    eval_config: str,
    out_dir: str,
    quick: bool,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    cfgs: dict[str, dict[str, Any]] = {
        "data.yaml": load_yaml(data_config),
        "iforest.yaml": load_yaml(iforest_config),
        "ppo.yaml": load_yaml(ppo_config),
        "eval.yaml": load_yaml(eval_config),
    }

    if quick:
        data_cfg = cfgs["data.yaml"]
        dataset_cfg = data_cfg.setdefault("dataset", {})
        dataset_cfg["limit"] = min(2000, int(dataset_cfg.get("limit", 2000)))
        if dataset_cfg.get("source") == "toy":
            dataset_cfg["n"] = min(1000, int(dataset_cfg.get("n", 5000)))

        iforest_cfg = cfgs["iforest.yaml"]
        iforest_cfg.setdefault("model", {})["n_estimators"] = min(
            50, int(iforest_cfg.get("model", {}).get("n_estimators", 200))
        )

        ppo_cfg = cfgs["ppo.yaml"]
        ppo_cfg.setdefault("train", {})["epochs"] = min(
            1, int(ppo_cfg.get("train", {}).get("epochs", 10))
        )
        ppo_cfg["train"]["batch_size"] = min(
            32, int(ppo_cfg.get("train", {}).get("batch_size", 64))
        )

    paths: dict[str, str] = {}
    for name, payload in cfgs.items():
        dest = os.path.join(out_dir, name)
        dump_yaml(dest, payload)
        paths[name] = dest

    return {
        "data": paths["data.yaml"],
        "iforest": paths["iforest.yaml"],
        "ppo": paths["ppo.yaml"],
        "eval": paths["eval.yaml"],
    }


def _load_contract_split(
    contract_dir: str, split: str
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int_]]:
    x7 = np.load(os.path.join(contract_dir, f"features_v7_{split}.npy")).astype(np.float32)
    x128 = np.load(os.path.join(contract_dir, f"features_v128_{split}.npy")).astype(np.float32)
    y = np.load(os.path.join(contract_dir, f"y_{split}.npy")).astype(np.int_)
    return x7, x128, y


def _load_contract_test(
    contract_dir: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int_]]:
    x7 = np.load(os.path.join(contract_dir, "features_v7_test.npy")).astype(np.float32)
    x128 = np.load(os.path.join(contract_dir, "features_v128_test.npy")).astype(np.float32)
    y = np.load(os.path.join(contract_dir, "y_true.npy")).astype(np.int_)
    return x7, x128, y


def _load_iforest_artifacts(model_dir: str) -> tuple[IForestModel, Preprocessor, float]:
    ifm = joblib.load(os.path.join(model_dir, "model.joblib"))
    prep = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    meta_path = os.path.join(model_dir, "meta.json")
    threshold = 0.5
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        threshold = float(meta.get("threshold", threshold))
        ifm.threshold = threshold
    return ifm, prep, threshold


def _load_ppo_actor(model_dir: str, ppo_cfg: dict[str, Any]) -> tuple[Actor, float]:
    actor = Actor(
        state_dim=ppo_cfg["rl"]["state_dim"],
        hidden=ppo_cfg["networks"]["hidden_sizes"],
        action_dim=ppo_cfg["rl"]["action_dim"],
    )
    actor_path = os.path.join(model_dir, "actor.pt")
    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    actor.eval()

    threshold = 0.5
    meta_path = os.path.join(model_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        threshold = float(meta.get("threshold", threshold))
    return actor, threshold


def _evaluate_from_contract(
    contract_dir: str,
    iforest_dir: str,
    ppo_dir: str,
    ppo_cfg: dict[str, Any],
    out_dir: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    os.makedirs(out_dir, exist_ok=True)
    x7_test, x128_test, y_true = _load_contract_test(contract_dir)

    ifm, prep, if_threshold = _load_iforest_artifacts(iforest_dir)
    x7p = prep.transform(x7_test)
    y_if_score = ifm.score(x7p)

    actor, rl_threshold = _load_ppo_actor(ppo_dir, ppo_cfg)
    y_rl_scores: list[float] = []
    with torch.no_grad():
        for row in x128_test:
            st = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            probs = actor(st).squeeze(0).numpy()
            y_rl_scores.append(float(probs[0] + probs[1]))
    y_rl_score = np.array(y_rl_scores, dtype=float)

    if_metrics = classification_metrics(y_true, y_if_score, threshold=if_threshold)
    rl_metrics = classification_metrics(y_true, y_rl_score, threshold=rl_threshold)

    metrics = {
        "n_test": int(len(y_true)),
        "iforest": if_metrics,
        "ppo": rl_metrics,
        "thresholds": {"iforest": if_threshold, "ppo": rl_threshold},
    }

    def cmatrix(scores: NDArray[np.float64], threshold: float) -> dict[str, int]:
        preds = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    conf = {
        "iforest": cmatrix(y_if_score, if_threshold),
        "ppo": cmatrix(y_rl_score, rl_threshold),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "confusion_matrix.json"), "w") as f:
        json.dump(conf, f, indent=2)

    return metrics, conf, {"iforest": if_threshold, "ppo": rl_threshold}


@app.command()
def main(
    data_config: str = "configs/data.yaml",
    iforest_config: str = "configs/iforest.yaml",
    ppo_config: str = "configs/ppo.yaml",
    eval_config: str = "configs/eval.yaml",
    out_root: str = "reports/runs",
    quick: bool = False,
    overwrite: bool = False,
    run_id: str | None = None,
) -> None:
    run_id = run_id or utc_run_id()
    run_root = os.path.join(out_root, run_id)
    if os.path.exists(run_root):
        if not overwrite:
            raise typer.BadParameter(f"Run already exists: {run_root} (use --overwrite)")
        shutil.rmtree(run_root)
    contract_dir = os.path.join(run_root, "contract")
    iforest_dir = os.path.join(run_root, "models", "iforest")
    ppo_dir = os.path.join(run_root, "models", "ppo")
    eval_dir = os.path.join(run_root, "eval")
    report_dir = os.path.join(run_root, "report")
    config_dir = os.path.join(run_root, "config")

    for path in (contract_dir, iforest_dir, ppo_dir, eval_dir, report_dir, config_dir):
        os.makedirs(path, exist_ok=True)

    config_paths = _snapshot_configs(
        data_config=data_config,
        iforest_config=iforest_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        out_dir=config_dir,
        quick=quick,
    )

    ppo_cfg = load_yaml(config_paths["ppo"])
    seed_value = int(ppo_cfg.get("rl", {}).get("seed", 42))
    set_global_seeds(seed_value)

    export_frozen_contract_to_dir(
        data_cfg_path=config_paths["data"],
        out_dir=contract_dir,
        include_splits=True,
        contract_id=run_id,
    )

    x7_train, x128_train, y_train = _load_contract_split(contract_dir, "train")
    x7_val, x128_val, y_val = _load_contract_split(contract_dir, "val")

    t0_if = time.perf_counter()
    train_iforest_from_arrays(config_paths["iforest"], x7_train, iforest_dir)
    t1_if = time.perf_counter()

    t0_ppo = time.perf_counter()
    train_ppo_from_arrays(
        config_paths["ppo"], x128_train, y_train, ppo_dir, x_val=x128_val, y_val=y_val
    )
    t1_ppo = time.perf_counter()

    metrics, _, thresholds = _evaluate_from_contract(
        contract_dir=contract_dir,
        iforest_dir=iforest_dir,
        ppo_dir=ppo_dir,
        ppo_cfg=ppo_cfg,
        out_dir=eval_dir,
    )

    schema_hash_path = os.path.join(contract_dir, "schema_hash.txt")
    with open(schema_hash_path, "r") as f:
        schema_hash = f.read().strip()

    contract_meta_path = os.path.join(contract_dir, "meta.json")
    with open(contract_meta_path, "r") as f:
        contract_meta = json.load(f)

    config_hashes = {
        "data": file_sha256(config_paths["data"]),
        "iforest": file_sha256(config_paths["iforest"]),
        "ppo": file_sha256(config_paths["ppo"]),
        "eval": file_sha256(config_paths["eval"]),
    }
    config_snapshot = {
        "data": os.path.relpath(config_paths["data"], run_root),
        "iforest": os.path.relpath(config_paths["iforest"], run_root),
        "ppo": os.path.relpath(config_paths["ppo"], run_root),
        "eval": os.path.relpath(config_paths["eval"], run_root),
    }

    poetry_lock_hash = None
    lock_path = os.path.join(os.getcwd(), "poetry.lock")
    if os.path.exists(lock_path):
        poetry_lock_hash = file_sha256(lock_path)

    iforest_cfg = load_yaml(config_paths["iforest"])
    iforest_seed = int(iforest_cfg.get("model", {}).get("random_state", seed_value))
    manifest = build_run_manifest(
        run_id=run_id,
        created_at=datetime.utcnow().isoformat() + "Z",
        git_commit=_git_commit_hash(),
        config_hashes=config_hashes,
        config_snapshot=config_snapshot,
        data_cfg=load_yaml(config_paths["data"]),
        schema_hash=schema_hash,
        poetry_lock_hash=poetry_lock_hash,
        contract_id=run_id,
        contract_meta=contract_meta,
        mode="quick" if quick else "full",
        seeds={
            "python": seed_value,
            "numpy": seed_value,
            "torch": seed_value,
            "iforest": iforest_seed,
        },
    )
    write_run_manifest(os.path.join(report_dir, "run_manifest.json"), manifest)

    summary_md = render_summary_md(
        run_id=run_id,
        git_commit=manifest["git_commit"],
        schema_hash=schema_hash,
        contract_meta=contract_meta,
        metrics={"iforest": metrics["iforest"], "ppo": metrics["ppo"]},
        thresholds=thresholds,
        train_times={"iforest": t1_if - t0_if, "ppo": t1_ppo - t0_ppo},
        iforest_cfg=iforest_cfg,
        ppo_cfg=ppo_cfg,
        contract_dir=contract_dir,
        run_root=run_root,
        mode="quick" if quick else "full",
    )
    with open(os.path.join(report_dir, "summary.md"), "w") as f:
        f.write(summary_md)

    typer.echo(f"Run complete -> {run_root}")


if __name__ == "__main__":
    app()
