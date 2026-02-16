from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from evaluation.morl_report import score_morl_weight_on_split

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_contract_test(contract_dir: Path) -> tuple[FloatArray, FloatArray, IntArray]:
    x7 = np.load(contract_dir / "features_v7_test.npy").astype(np.float64)
    x128 = np.load(contract_dir / "features_v128_test.npy").astype(np.float64)
    y_true = np.load(contract_dir / "y_true.npy").astype(np.int_)
    return x7, x128, y_true


def _score_iforest(model_dir: Path, x7: FloatArray) -> FloatArray:
    ifm = joblib.load(model_dir / "model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    x7p = scaler.transform(x7)
    score = ifm.score(x7p)
    return np.asarray(score, dtype=np.float64)


def _metrics_at_k(y_true: IntArray, scores: FloatArray, k: int) -> dict[str, float]:
    mask = y_true != -1
    y = y_true[mask]
    s = scores[mask]
    n = int(y.shape[0])
    if n == 0:
        return {
            "n_eval": 0.0,
            "alerts_per_1k": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "tp": float("nan"),
            "fp": float("nan"),
            "fn": float("nan"),
            "tn": float("nan"),
        }

    k_eff = max(0, min(int(k), n))
    order = np.argsort(-s, kind="mergesort")
    pred = np.zeros(n, dtype=np.int_)
    if k_eff > 0:
        pred[order[:k_eff]] = 1

    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    if np.unique(y).shape[0] < 2:
        roc = float("nan")
        pr = float("nan")
    else:
        try:
            roc = float(roc_auc_score(y, s))
        except Exception:
            roc = float("nan")
        try:
            pr = float(average_precision_score(y, s))
        except Exception:
            pr = float("nan")

    tp = float(np.sum((y == 1) & (pred == 1)))
    fp = float(np.sum((y == 0) & (pred == 1)))
    fn = float(np.sum((y == 1) & (pred == 0)))
    tn = float(np.sum((y == 0) & (pred == 0)))

    return {
        "n_eval": float(n),
        "alerts_per_1k": float(1000.0 * k_eff / n),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": roc,
        "pr_auc": pr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _parse_k_values(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        v = token.strip()
        if not v:
            continue
        out.append(int(v))
    if not out:
        raise ValueError("No K values provided")
    return out


def _fmt(v: float, digits: int = 3) -> str:
    if not math.isfinite(v):
        return "N/A"
    return f"{v:.{digits}f}"


def run_k_sweep(run_dir: Path, k_values: list[int]) -> list[dict[str, Any]]:
    contract_dir = run_dir / "contract"
    x7_test, _, y_test = _load_contract_test(contract_dir)

    iforest_scores = _score_iforest(run_dir / "models" / "iforest", x7_test)

    selection = _load_json(run_dir / "eval" / "morl" / "meta_selection.json")
    selected_weight_raw = selection.get("selected_weight")
    if not isinstance(selected_weight_raw, list):
        raise ValueError("meta_selection.json missing selected_weight")
    selected_weight = [float(v) for v in selected_weight_raw]

    y_morl, morl_scores = score_morl_weight_on_split(
        contract_dir=str(contract_dir),
        morl_model_dir=str(run_dir / "models" / "morl"),
        morl_cfg_path=str(run_dir / "config" / "morl.yaml"),
        split="test",
        w=selected_weight,
    )
    if y_morl.shape[0] != y_test.shape[0]:
        raise ValueError("MORL and contract y lengths do not match")

    outputs: list[dict[str, Any]] = []
    for model_name, score in (("morl_selected", morl_scores), ("iforest", iforest_scores)):
        for k in k_values:
            mm = _metrics_at_k(y_test, score, k)
            outputs.append(
                {
                    "run_id": run_dir.name,
                    "model": model_name,
                    "k": int(k),
                    "n_eval": int(mm["n_eval"]) if math.isfinite(mm["n_eval"]) else "",
                    "alerts_per_1k": mm["alerts_per_1k"],
                    "precision": mm["precision"],
                    "recall": mm["recall"],
                    "f1": mm["f1"],
                    "roc_auc": mm["roc_auc"],
                    "pr_auc": mm["pr_auc"],
                    "tp": int(mm["tp"]) if math.isfinite(mm["tp"]) else "",
                    "fp": int(mm["fp"]) if math.isfinite(mm["fp"]) else "",
                    "fn": int(mm["fn"]) if math.isfinite(mm["fn"]) else "",
                    "tn": int(mm["tn"]) if math.isfinite(mm["tn"]) else "",
                }
            )
    return outputs


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "model",
        "k",
        "n_eval",
        "alerts_per_1k",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# K-Sweep Summary",
        "",
        "| model | K | alerts/1k | precision | recall | f1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['k']} | {_fmt(float(row['alerts_per_1k']))} | "
            f"{_fmt(float(row['precision']))} | {_fmt(float(row['recall']))} | {_fmt(float(row['f1']))} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def write_tex(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Model & K & Alerts/1k & Precision & Recall & F1 \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['k']} & {_fmt(float(row['alerts_per_1k']))} & "
            f"{_fmt(float(row['precision']))} & {_fmt(float(row['recall']))} & {_fmt(float(row['f1']))} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build K-sweep tables from a run directory")
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Run directory (reports/runs/<run_id>)"
    )
    parser.add_argument(
        "--k-values", default="50,100,200", help="Comma-separated K values (e.g. 50,100,200)"
    )
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--out-md", type=Path, required=True, help="Output markdown path")
    parser.add_argument("--out-tex", type=Path, required=True, help="Output LaTeX path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = _parse_k_values(args.k_values)
    rows = run_k_sweep(args.run_dir, k_values)
    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)
    write_tex(rows, args.out_tex)
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_md}")
    print(f"Wrote: {args.out_tex}")


if __name__ == "__main__":
    main()
