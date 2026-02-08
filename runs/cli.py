from __future__ import annotations

import json
import os
import random
import subprocess
import time
import shutil
from datetime import datetime
from typing import Any, cast

import joblib
import numpy as np
import torch
import typer
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from architecture_a_rl.networks import Actor
from architecture_a_rl.morl.moppo import load_weight_grid_from_config, train_moppo_from_arrays
from architecture_a_rl.morl.meta_controller import select_weight
from architecture_a_rl.train import train_ppo_from_arrays
from architecture_b_iforest.model import IForestModel
from architecture_b_iforest.preprocess import Preprocessor
from architecture_b_iforest.train import train_iforest_from_arrays
from data.contract_export import export_frozen_contract_to_dir
from data.join import build_join_map
from evaluation.metrics import classification_metrics
from evaluation.meta_selection_report import write_meta_selection_artifacts
from evaluation.morl_report import evaluate_morl_weight_on_split, run_morl_weight_sweep
from evaluation.policy_eval import run_policy_eval
from evaluation.robustness import run_robustness_suite
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
    robustness_config: str | None = None,
    morl_config: str | None = None,
    meta_config: str | None = None,
    join_config: str | None = None,
    policy_eval_config: str | None = None,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    cfgs: dict[str, dict[str, Any]] = {
        "data.yaml": load_yaml(data_config),
        "iforest.yaml": load_yaml(iforest_config),
        "ppo.yaml": load_yaml(ppo_config),
        "eval.yaml": load_yaml(eval_config),
    }
    if robustness_config:
        cfgs["robustness.yaml"] = load_yaml(robustness_config)
    if morl_config:
        cfgs["morl.yaml"] = load_yaml(morl_config)
    if meta_config:
        cfgs["meta_controller.yaml"] = load_yaml(meta_config)
    if join_config:
        cfgs["join.yaml"] = load_yaml(join_config)
    if policy_eval_config:
        cfgs["policy_eval.yaml"] = load_yaml(policy_eval_config)

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

        if "morl.yaml" in cfgs:
            morl_cfg = cfgs["morl.yaml"].setdefault("morl", {})
            train_cfg = morl_cfg.setdefault("training", {})
            train_cfg["epochs"] = min(1, int(train_cfg.get("epochs", 5)))
            train_cfg["batch_size"] = min(64, int(train_cfg.get("batch_size", 256)))

    paths: dict[str, str] = {}
    for name, payload in cfgs.items():
        dest = os.path.join(out_dir, name)
        dump_yaml(dest, payload)
        paths[name] = dest

    out_paths: dict[str, str] = {
        "data": paths["data.yaml"],
        "iforest": paths["iforest.yaml"],
        "ppo": paths["ppo.yaml"],
        "eval": paths["eval.yaml"],
    }
    if "robustness.yaml" in paths:
        out_paths["robustness"] = paths["robustness.yaml"]
    if "morl.yaml" in paths:
        out_paths["morl"] = paths["morl.yaml"]
    if "meta_controller.yaml" in paths:
        out_paths["meta"] = paths["meta_controller.yaml"]
    if "join.yaml" in paths:
        out_paths["join"] = paths["join.yaml"]
    if "policy_eval.yaml" in paths:
        out_paths["policy_eval"] = paths["policy_eval.yaml"]
    return out_paths


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


def _score_iforest(model_dir: str, x7: NDArray[np.float32]) -> tuple[NDArray[np.float64], float]:
    ifm, prep, threshold = _load_iforest_artifacts(model_dir)
    x7p = prep.transform(x7)
    scores = ifm.score(x7p).astype(np.float64)
    return scores, threshold


def _score_ppo(
    model_dir: str, ppo_cfg: dict[str, Any], x128: NDArray[np.float32]
) -> tuple[NDArray[np.float64], float]:
    actor, threshold = _load_ppo_actor(model_dir, ppo_cfg)
    y_rl_scores: list[float] = []
    with torch.no_grad():
        for row in x128:
            st = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            probs = actor(st).squeeze(0).numpy()
            y_rl_scores.append(float(probs[0] + probs[1]))
    return np.asarray(y_rl_scores, dtype=np.float64), threshold


def _evaluate_from_contract(
    contract_dir: str,
    iforest_dir: str,
    ppo_dir: str,
    ppo_cfg: dict[str, Any],
    out_dir: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    os.makedirs(out_dir, exist_ok=True)
    x7_test, x128_test, y_true = _load_contract_test(contract_dir)

    y_if_score, if_threshold = _score_iforest(iforest_dir, x7_test)
    y_rl_score, rl_threshold = _score_ppo(ppo_dir, ppo_cfg, x128_test)

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
    robustness_config: str | None = None,
    morl_config: str | None = None,
    meta_config: str | None = None,
    join_config: str | None = None,
    policy_eval_config: str | None = None,
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
    morl_dir = os.path.join(run_root, "models", "morl")
    eval_dir = os.path.join(run_root, "eval")
    report_dir = os.path.join(run_root, "report")
    config_dir = os.path.join(run_root, "config")

    for path in (contract_dir, iforest_dir, ppo_dir, morl_dir, eval_dir, report_dir, config_dir):
        os.makedirs(path, exist_ok=True)

    config_paths = _snapshot_configs(
        data_config=data_config,
        iforest_config=iforest_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        out_dir=config_dir,
        quick=quick,
        robustness_config=robustness_config,
        morl_config=morl_config,
        meta_config=meta_config,
        join_config=join_config,
        policy_eval_config=policy_eval_config,
    )

    ppo_cfg = load_yaml(config_paths["ppo"])
    seed_value = int(ppo_cfg.get("rl", {}).get("seed", 42))
    set_global_seeds(seed_value)
    join_meta: dict[str, Any] = {}
    join_map_path: str | None = None
    join_meta_path: str | None = None
    policy_eval_summary: dict[str, Any] = {}
    policy_eval_json_path: str | None = None
    policy_eval_md_path: str | None = None

    export_frozen_contract_to_dir(
        data_cfg_path=config_paths["data"],
        out_dir=contract_dir,
        include_splits=True,
        contract_id=run_id,
    )

    if config_paths.get("join"):
        join_out_dir = os.path.join(eval_dir, "join")
        join_result = build_join_map(
            contract_dir=contract_dir,
            join_cfg_path=config_paths["join"],
            out_dir=join_out_dir,
        )
        join_meta = cast(dict[str, Any], join_result.get("meta", {}))
        join_map_path = cast(str | None, join_result.get("join_map"))
        join_meta_path = cast(str | None, join_result.get("join_meta"))

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

    morl_enabled = False
    morl_hash: str | None = None
    morl_weight_grid: list[list[float]] = []
    t0_morl: float | None = None
    t1_morl: float | None = None
    meta_enabled = False
    meta_hash: str | None = None
    selected_weight: list[float] | None = None
    selected_test_path: str | None = None
    selected_test_md_path: str | None = None
    meta_method: dict[str, Any] = {}
    meta_constraints: dict[str, Any] = {}
    meta_selection_path: str | None = None
    meta_selection_md_path: str | None = None
    if config_paths.get("morl"):
        morl_cfg = load_yaml(config_paths["morl"])
        morl_enabled = bool(morl_cfg.get("morl", {}).get("enabled", False))
        morl_hash = file_sha256(config_paths["morl"])
        if morl_enabled:
            t0_morl = time.perf_counter()
            train_moppo_from_arrays(
                config_paths["morl"],
                x128_train,
                y_train,
                morl_dir,
                seed=seed_value,
                x_val=x128_val,
                y_val=y_val,
                contract_dir=contract_dir,
            )
            t1_morl = time.perf_counter()
            grid = load_weight_grid_from_config(config_paths["morl"])
            morl_weight_grid = [[float(v) for v in row.tolist()] for row in grid]
            morl_eval_dir = os.path.join(eval_dir, "morl")
            run_morl_weight_sweep(
                contract_dir=contract_dir,
                morl_model_dir=morl_dir,
                morl_cfg_path=config_paths["morl"],
                out_dir=morl_eval_dir,
                split="val",
            )
            run_morl_weight_sweep(
                contract_dir=contract_dir,
                morl_model_dir=morl_dir,
                morl_cfg_path=config_paths["morl"],
                out_dir=morl_eval_dir,
                split="test",
            )

            if config_paths.get("meta"):
                meta_cfg = load_yaml(config_paths["meta"])
                meta_enabled = bool(meta_cfg.get("meta_controller", {}).get("enabled", False))
                meta_hash = file_sha256(config_paths["meta"])
                if meta_enabled:
                    selection = select_weight(
                        meta_cfg_path=config_paths["meta"],
                        morl_cfg_path=config_paths["morl"],
                        val_results_path=os.path.join(morl_eval_dir, "morl_results_val.json"),
                        seed=seed_value,
                    )
                    selected_weight = [float(v) for v in selection["selected_weight"]]
                    meta_method = cast(dict[str, Any], selection.get("method", {}))
                    meta_constraints = cast(dict[str, Any], selection.get("constraints", {}))
                    selected_row = evaluate_morl_weight_on_split(
                        contract_dir=contract_dir,
                        morl_model_dir=morl_dir,
                        morl_cfg_path=config_paths["morl"],
                        split="test",
                        w=selected_weight,
                    )
                    artifacts = write_meta_selection_artifacts(
                        out_dir=morl_eval_dir,
                        selection=selection,
                        selected_test=selected_row,
                    )
                    selected_test_path = artifacts["selected_test_json"]
                    selected_test_md_path = artifacts["selected_test_md"]
                    meta_selection_path = artifacts["meta_selection_json"]
                    meta_selection_md_path = artifacts["meta_selection_md"]

    morl_objective_source: str | None = None
    morl_level1_stats: dict[str, float] = {}
    morl_normalization_summary: dict[str, Any] = {}
    morl_meta_path = os.path.join(morl_dir, "morl_meta.json")
    if os.path.exists(morl_meta_path):
        with open(morl_meta_path, "r") as f:
            morl_meta_payload = cast(dict[str, Any], json.load(f))
        source_raw = morl_meta_payload.get("objective_source")
        if isinstance(source_raw, str):
            morl_objective_source = source_raw
        stats_raw = morl_meta_payload.get("realdata_objective_stats", {})
        if isinstance(stats_raw, dict):
            morl_level1_stats = {str(k): float(v) for k, v in stats_raw.items()}
        norm_raw = morl_meta_payload.get("normalization_summary", {})
        if isinstance(norm_raw, dict):
            morl_normalization_summary = cast(dict[str, Any], norm_raw)

    metrics, _, thresholds = _evaluate_from_contract(
        contract_dir=contract_dir,
        iforest_dir=iforest_dir,
        ppo_dir=ppo_dir,
        ppo_cfg=ppo_cfg,
        out_dir=eval_dir,
    )

    if config_paths.get("policy_eval"):
        x7_val_pe, x128_val_pe, y_val_pe = _load_contract_split(contract_dir, "val")
        x7_test_pe, x128_test_pe, y_test_pe = _load_contract_test(contract_dir)
        if_scores_val, _ = _score_iforest(iforest_dir, x7_val_pe)
        if_scores_test, _ = _score_iforest(iforest_dir, x7_test_pe)
        ppo_scores_val, _ = _score_ppo(ppo_dir, ppo_cfg, x128_val_pe)
        ppo_scores_test, _ = _score_ppo(ppo_dir, ppo_cfg, x128_test_pe)
        policy_eval_out = run_policy_eval(
            policy_cfg_path=config_paths["policy_eval"],
            y_val=y_val_pe,
            y_test=y_test_pe,
            val_scores_by_model={"iforest": if_scores_val, "ppo": ppo_scores_val},
            test_scores_by_model={"iforest": if_scores_test, "ppo": ppo_scores_test},
            out_dir=os.path.join(eval_dir, "policy"),
        )
        policy_eval_summary = {
            "mode": policy_eval_out.get("mode"),
            "primary_metric": policy_eval_out.get("primary_metric"),
            "constraints": policy_eval_out.get("constraints"),
        }
        paths_raw = cast(dict[str, Any], policy_eval_out.get("paths", {}))
        policy_eval_json_path = cast(str | None, paths_raw.get("json"))
        policy_eval_md_path = cast(str | None, paths_raw.get("md"))

    robustness_hash = None
    robustness_variants: list[str] = []
    robustness_enabled = False
    if config_paths.get("robustness"):
        robustness_cfg = load_yaml(config_paths["robustness"])
        rcfg = robustness_cfg.get("robustness", {})
        robustness_enabled = bool(rcfg.get("enabled", True))
        robustness_variants = [str(v.get("name", "")) for v in rcfg.get("variants", [])]
        robustness_hash = file_sha256(config_paths["robustness"])

        if robustness_enabled:
            robustness_dir = os.path.join(eval_dir, "robustness")
            run_robustness_suite(
                contract_dir=contract_dir,
                iforest_model_dir=iforest_dir,
                ppo_model_dir=ppo_dir,
                ppo_config=config_paths["ppo"],
                robustness_cfg_path=config_paths["robustness"],
                out_dir=robustness_dir,
                quick=quick,
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
    if config_paths.get("robustness"):
        config_hashes["robustness"] = file_sha256(config_paths["robustness"])
    if config_paths.get("morl"):
        config_hashes["morl"] = file_sha256(config_paths["morl"])
    if config_paths.get("meta"):
        config_hashes["meta_controller"] = file_sha256(config_paths["meta"])
    if config_paths.get("join"):
        config_hashes["join"] = file_sha256(config_paths["join"])
    if config_paths.get("policy_eval"):
        config_hashes["policy_eval"] = file_sha256(config_paths["policy_eval"])
    config_snapshot = {
        "data": os.path.relpath(config_paths["data"], run_root),
        "iforest": os.path.relpath(config_paths["iforest"], run_root),
        "ppo": os.path.relpath(config_paths["ppo"], run_root),
        "eval": os.path.relpath(config_paths["eval"], run_root),
    }
    if config_paths.get("robustness"):
        config_snapshot["robustness"] = os.path.relpath(config_paths["robustness"], run_root)
    if config_paths.get("morl"):
        config_snapshot["morl"] = os.path.relpath(config_paths["morl"], run_root)
    if config_paths.get("meta"):
        config_snapshot["meta_controller"] = os.path.relpath(config_paths["meta"], run_root)
    if config_paths.get("join"):
        config_snapshot["join"] = os.path.relpath(config_paths["join"], run_root)
    if config_paths.get("policy_eval"):
        config_snapshot["policy_eval"] = os.path.relpath(config_paths["policy_eval"], run_root)

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
        robustness={
            "enabled": robustness_enabled,
            "config_sha256": robustness_hash,
            "variants": robustness_variants,
        },
        morl={
            "enabled": morl_enabled,
            "config_sha256": morl_hash,
            "weight_grid": morl_weight_grid,
            "model_dir": os.path.relpath(morl_dir, run_root) if morl_enabled else None,
            "objective_source": morl_objective_source,
            "realdata_objective_stats": morl_level1_stats,
            "normalization_summary": morl_normalization_summary,
            "train_seconds": (t1_morl - t0_morl)
            if (t0_morl is not None and t1_morl is not None)
            else None,
        },
        meta_controller={
            "enabled": meta_enabled,
            "meta_config_hash": meta_hash,
            "selected_weight": selected_weight,
            "selection_method": meta_method,
            "constraints": meta_constraints,
            "selection_output_path": os.path.relpath(meta_selection_path, run_root)
            if meta_selection_path
            else None,
            "selection_report_path": os.path.relpath(meta_selection_md_path, run_root)
            if meta_selection_md_path
            else None,
            "selected_test_output_path": os.path.relpath(selected_test_path, run_root)
            if selected_test_path
            else None,
            "selected_test_report_path": os.path.relpath(selected_test_md_path, run_root)
            if selected_test_md_path
            else None,
        },
        join_diagnostics={
            "enabled": bool(config_paths.get("join")),
            "config_sha256": file_sha256(config_paths["join"])
            if config_paths.get("join")
            else None,
            "join_map_path": os.path.relpath(join_map_path, run_root) if join_map_path else None,
            "join_meta_path": os.path.relpath(join_meta_path, run_root) if join_meta_path else None,
            "summary": join_meta,
        },
        policy_eval={
            "enabled": bool(config_paths.get("policy_eval")),
            "config_sha256": file_sha256(config_paths["policy_eval"])
            if config_paths.get("policy_eval")
            else None,
            "summary": policy_eval_summary,
            "json_path": os.path.relpath(policy_eval_json_path, run_root)
            if policy_eval_json_path
            else None,
            "report_path": os.path.relpath(policy_eval_md_path, run_root)
            if policy_eval_md_path
            else None,
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
        policy_eval=policy_eval_summary if policy_eval_summary else None,
    )
    with open(os.path.join(report_dir, "summary.md"), "w") as f:
        f.write(summary_md)

    typer.echo(f"Run complete -> {run_root}")


if __name__ == "__main__":
    app()
