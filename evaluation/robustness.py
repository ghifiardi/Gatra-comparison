from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import torch
import typer
import yaml
from numpy.typing import NDArray

from architecture_a_rl.networks import Actor
from architecture_b_iforest.model import IForestModel
from architecture_b_iforest.preprocess import Preprocessor
from evaluation.metrics import classification_metrics
from evaluation.label_availability import (
    apply_label_availability,
    write_label_availability_artifacts,
)
from evaluation.variants import (
    VariantBatch,
    apply_label_delay,
    apply_missingness,
    apply_noise,
    apply_time_slice,
)


FloatArr = NDArray[np.float32]
IntArr = NDArray[np.int8]
MissingStrategy = Literal["mcar", "topk"]
MissingFill = Literal["zero", "mean", "median"]
NoiseDistribution = Literal["gaussian", "laplace"]
LabelDelayPolicy = Literal["treat_as_benign", "treat_as_unknown", "drop"]
AvailabilityPolicy = Literal["treat_as_benign", "treat_as_unknown"]

app = typer.Typer()


def _read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return cast(dict[str, Any], payload)


def _as_missing_strategy(value: str) -> MissingStrategy:
    if value not in ("mcar", "topk"):
        raise ValueError(f"Unknown missingness strategy: {value}")
    return cast(MissingStrategy, value)


def _as_missing_fill(value: str) -> MissingFill:
    if value not in ("zero", "mean", "median"):
        raise ValueError(f"Unknown missingness fill: {value}")
    return cast(MissingFill, value)


def _as_noise_distribution(value: str) -> NoiseDistribution:
    if value not in ("gaussian", "laplace"):
        raise ValueError(f"Unknown noise distribution: {value}")
    return cast(NoiseDistribution, value)


def _as_label_policy(value: str) -> LabelDelayPolicy:
    if value not in ("treat_as_benign", "treat_as_unknown", "drop"):
        raise ValueError(f"Unknown label delay policy: {value}")
    return cast(LabelDelayPolicy, value)


def _load_contract_test(
    contract_dir: str,
) -> tuple[FloatArr, FloatArr, IntArr, NDArray[np.int64], NDArray[np.int64]]:
    p = Path(contract_dir)
    X7 = np.load(p / "features_v7_test.npy").astype(np.float32)
    X128 = np.load(p / "features_v128_test.npy").astype(np.float32)
    y_path = p / "y_test.npy"
    if y_path.exists():
        y = np.load(y_path)
    else:
        y = np.load(p / "y_true.npy")
    timestamps = np.load(p / "timestamps_epoch_s_test.npy").astype(np.int64)
    created_path = p / "label_created_at_epoch_s_test.npy"
    if created_path.exists():
        label_created = np.load(created_path).astype(np.int64)
    else:
        label_created = timestamps.copy()
    if y.shape[0] != timestamps.shape[0] or y.shape[0] != label_created.shape[0]:
        raise ValueError("Contract label/timestamp arrays are misaligned for robustness evaluation")
    return X7, X128, y.astype(np.int8), timestamps, label_created


def _filter_unknown(y: IntArr, scores: NDArray[np.floating[Any]]) -> tuple[IntArr, FloatArr]:
    mask = y != -1
    return y[mask].astype(np.int8), scores[mask].astype(np.float32)


def _load_iforest_artifacts(model_dir: str) -> tuple[IForestModel, Preprocessor, float]:
    ifm: IForestModel = joblib.load(Path(model_dir) / "model.joblib")
    prep: Preprocessor = joblib.load(Path(model_dir) / "scaler.joblib")
    meta_path = Path(model_dir) / "meta.json"
    threshold = 0.5
    if meta_path.exists():
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
    actor_path = Path(model_dir) / "actor.pt"
    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    actor.eval()

    threshold = 0.5
    meta_path = Path(model_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        threshold = float(meta.get("threshold", threshold))
    return actor, threshold


def _evaluate_models(
    if_model_dir: str,
    ppo_model_dir: str,
    ppo_cfg: dict[str, Any],
    X7: FloatArr,
    X128: FloatArr,
    y: IntArr,
) -> tuple[dict[str, Any], dict[str, float]]:
    ifm, prep, if_threshold = _load_iforest_artifacts(if_model_dir)
    x7p = prep.transform(X7)
    if_scores = ifm.score(x7p).astype(np.float32)

    actor, rl_threshold = _load_ppo_actor(ppo_model_dir, ppo_cfg)
    scores: list[float] = []
    with torch.no_grad():
        for row in X128:
            st = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            probs = actor(st).squeeze(0).numpy()
            scores.append(float(probs[0] + probs[1]))
    rl_scores = np.array(scores, dtype=np.float32)

    y_if, if_scores_f = _filter_unknown(y, if_scores)
    y_rl, rl_scores_f = _filter_unknown(y, rl_scores)

    if_metrics = classification_metrics(
        y_if.astype(np.int_), if_scores_f.astype(np.float64), if_threshold
    )
    rl_metrics = classification_metrics(
        y_rl.astype(np.int_), rl_scores_f.astype(np.float64), rl_threshold
    )

    if_metrics["threshold"] = float(if_threshold)
    rl_metrics["threshold"] = float(rl_threshold)
    return {"iforest": if_metrics, "ppo": rl_metrics}, {
        "iforest": if_threshold,
        "ppo": rl_threshold,
    }


def run_robustness_suite(
    contract_dir: str,
    iforest_model_dir: str,
    ppo_model_dir: str,
    ppo_config: str,
    robustness_cfg_path: str,
    out_dir: str,
    *,
    quick: bool = False,
) -> None:
    cfg = _read_yaml(robustness_cfg_path)
    rcfg = cfg.get("robustness", {})
    seed = int(rcfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    X7_base, X128_base, y_base, ts_base, label_created_base = _load_contract_test(contract_dir)
    ppo_cfg = _read_yaml(ppo_config)

    variants = rcfg.get("variants", [])
    if quick:
        maxv = int(rcfg.get("quick", {}).get("max_variants", 3))
        variants = variants[:maxv]

    results: list[dict[str, Any]] = []

    baseline_metrics, thresholds = _evaluate_models(
        iforest_model_dir, ppo_model_dir, ppo_cfg, X7_base, X128_base, y_base
    )
    results.append({"variant": "baseline", "kind": "none", "meta": {}, "metrics": baseline_metrics})

    for v in variants:
        name = str(v["name"])
        kind = str(v["kind"])

        if name == "baseline":
            continue

        batches: list[VariantBatch] = []

        if kind == "none":
            batches = [VariantBatch(name=name, X7=X7_base, X128=X128_base, y=y_base, meta={})]

        elif kind == "missingness":
            rate = float(v["rate"])
            strategy = _as_missing_strategy(str(v.get("strategy", "mcar")))
            fill = _as_missing_fill(str(v.get("fill", "zero")))
            X7 = apply_missingness(X7_base, rate, strategy, fill, rng)
            X128 = apply_missingness(X128_base, rate, strategy, fill, rng)
            batches = [VariantBatch(name=name, X7=X7, X128=X128, y=y_base, meta={"missingness": v})]

        elif kind == "noise":
            sigma = float(v["sigma"])
            dist = _as_noise_distribution(str(v.get("distribution", "gaussian")))
            clamp = bool(v.get("clamp", True))
            X7 = apply_noise(X7_base, sigma, dist, clamp, rng)
            X128 = apply_noise(X128_base, sigma, dist, clamp, rng)
            batches = [VariantBatch(name=name, X7=X7, X128=X128, y=y_base, meta={"noise": v})]

        elif kind == "time_slice":
            slices = v.get("slices", [])
            for s in slices:
                label = str(s["label"])
                fs = float(s["frac_start"])
                fe = float(s["frac_end"])
                batches.append(apply_time_slice(X7_base, X128_base, y_base, fs, fe, label))

        elif kind == "label_delay":
            policy = _as_label_policy(str(v.get("policy", "treat_as_unknown")))
            if "delay" in v or "eval_time_epoch_s" in v:
                resolved_policy = (
                    "treat_as_unknown" if policy == "drop" else cast(AvailabilityPolicy, policy)
                )
                y2, available_mask, meta_delay = apply_label_availability(
                    y_true=y_base.astype(np.int_),
                    event_timestamps_epoch_s=ts_base,
                    label_created_at_epoch_s=label_created_base,
                    delay=str(v.get("delay", "7d")),
                    policy=resolved_policy,
                    eval_time_epoch_s=int(v["eval_time_epoch_s"])
                    if "eval_time_epoch_s" in v
                    else None,
                )
                write_label_availability_artifacts(
                    out_dir=str(outp),
                    split="test",
                    available_mask=available_mask,
                    y_available=y2,
                    meta=meta_delay,
                )
            else:
                frac = float(v.get("fraction", 0.1))
                y2, meta_delay = apply_label_delay(y_base, frac, policy, rng)
            batches = [VariantBatch(name=name, X7=X7_base, X128=X128_base, y=y2, meta=meta_delay)]

        elif kind == "compose":
            steps = v.get("steps", [])
            X7 = X7_base.copy()
            X128 = X128_base.copy()
            y = y_base.copy()
            meta_compose: dict[str, Any] = {"compose": steps}

            for step in steps:
                sk = str(step["kind"])
                if sk == "missingness":
                    X7 = apply_missingness(
                        X7,
                        float(step["rate"]),
                        _as_missing_strategy(str(step.get("strategy", "mcar"))),
                        _as_missing_fill(str(step.get("fill", "zero"))),
                        rng,
                    )
                    X128 = apply_missingness(
                        X128,
                        float(step["rate"]),
                        _as_missing_strategy(str(step.get("strategy", "mcar"))),
                        _as_missing_fill(str(step.get("fill", "zero"))),
                        rng,
                    )
                elif sk == "noise":
                    X7 = apply_noise(
                        X7,
                        float(step["sigma"]),
                        _as_noise_distribution(str(step.get("distribution", "gaussian"))),
                        bool(step.get("clamp", True)),
                        rng,
                    )
                    X128 = apply_noise(
                        X128,
                        float(step["sigma"]),
                        _as_noise_distribution(str(step.get("distribution", "gaussian"))),
                        bool(step.get("clamp", True)),
                        rng,
                    )
                elif sk == "label_delay":
                    policy = _as_label_policy(str(step.get("policy", "treat_as_unknown")))
                    if "delay" in step or "eval_time_epoch_s" in step:
                        resolved_policy = (
                            "treat_as_unknown"
                            if policy == "drop"
                            else cast(AvailabilityPolicy, policy)
                        )
                        y, available_mask, meta_delay = apply_label_availability(
                            y_true=y.astype(np.int_),
                            event_timestamps_epoch_s=ts_base,
                            label_created_at_epoch_s=label_created_base,
                            delay=str(step.get("delay", "7d")),
                            policy=resolved_policy,
                            eval_time_epoch_s=int(step["eval_time_epoch_s"])
                            if "eval_time_epoch_s" in step
                            else None,
                        )
                        write_label_availability_artifacts(
                            out_dir=str(outp),
                            split="test",
                            available_mask=available_mask,
                            y_available=y,
                            meta=meta_delay,
                        )
                    else:
                        y, _m = apply_label_delay(
                            y,
                            float(step.get("fraction", 0.1)),
                            policy,
                            rng,
                        )
                else:
                    raise ValueError(f"Unknown compose step kind: {sk}")

            batches = [VariantBatch(name=name, X7=X7, X128=X128, y=y, meta=meta_compose)]

        else:
            raise ValueError(f"Unknown robustness variant kind: {kind}")

        for b in batches:
            m, _ = _evaluate_models(iforest_model_dir, ppo_model_dir, ppo_cfg, b.X7, b.X128, b.y)
            results.append({"variant": b.name, "kind": kind, "meta": b.meta, "metrics": m})

    (outp / "robustness_results.json").write_text(json.dumps(results, indent=2))

    base = results[0]["metrics"]
    deltas: list[dict[str, Any]] = []
    for r in results[1:]:
        row = {"variant": r["variant"]}
        for model in ("iforest", "ppo"):
            row[f"{model}.delta_pr_auc"] = float(
                r["metrics"][model]["pr_auc"] - base[model]["pr_auc"]
            )
            row[f"{model}.delta_f1"] = float(r["metrics"][model]["f1"] - base[model]["f1"])
        deltas.append(row)

    (outp / "robustness_deltas.json").write_text(json.dumps(deltas, indent=2))

    lines = ["variant,model,roc_auc,pr_auc,precision,recall,f1,threshold"]
    for r in results:
        for model in ("iforest", "ppo"):
            mm = r["metrics"][model]
            lines.append(
                f"{r['variant']},{model},{mm['roc_auc']:.6f},{mm['pr_auc']:.6f},"
                f"{mm['precision']:.6f},{mm['recall']:.6f},{mm['f1']:.6f},{mm['threshold']:.6f}"
            )
    (outp / "robustness_table.csv").write_text("\n".join(lines) + "\n")

    md = []
    md.append("# Robustness Report\n")
    md.append(f"- Seed: {seed}\n")
    md.append(f"- Contract dir: {contract_dir}\n\n")
    md.append("## Variants\n")
    for r in results:
        md.append(f"- {r['variant']}\n")
    md.append("\n## Key deltas vs baseline (PR-AUC, F1)\n")
    md.append("See `robustness_deltas.json`.\n")
    (outp / "robustness.md").write_text("".join(md))


@app.command()
def main(
    contract_dir: str = typer.Option(..., "--contract-dir"),
    iforest_model_dir: str = typer.Option(..., "--iforest-model-dir"),
    ppo_model_dir: str = typer.Option(..., "--ppo-model-dir"),
    ppo_config: str = typer.Option(..., "--ppo-config"),
    robustness_config: str = "configs/robustness.yaml",
    out_dir: str = "reports/robustness",
    quick: bool = False,
) -> None:
    run_robustness_suite(
        contract_dir=contract_dir,
        iforest_model_dir=iforest_model_dir,
        ppo_model_dir=ppo_model_dir,
        ppo_config=ppo_config,
        robustness_cfg_path=robustness_config,
        out_dir=out_dir,
        quick=quick,
    )


if __name__ == "__main__":
    app()
