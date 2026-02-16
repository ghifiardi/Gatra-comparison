from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return float("nan")
        if v.lower() in {"nan", "none", "null", "n/a"}:
            return float("nan")
        try:
            return float(v)
        except ValueError:
            return float("nan")
    return float("nan")


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _load_row(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {str(k): str(v) for k, v in row.items()}
    raise ValueError(f"No rows found in {path}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def build_paragraph(
    row: dict[str, str],
    stats: dict[str, Any],
) -> str:
    f1_morl = _to_float(row.get("morl_f1"))
    f1_if = _to_float(row.get("policy_iforest_f1"))
    fpr_proxy_morl = _to_float(row.get("morl_alerts_per_1k"))
    fpr_proxy_if = _to_float(row.get("policy_iforest_alerts_per_1k"))

    delta_f1 = f1_morl - f1_if if math.isfinite(f1_morl) and math.isfinite(f1_if) else float("nan")
    delta_alerts = (
        fpr_proxy_morl - fpr_proxy_if
        if math.isfinite(fpr_proxy_morl) and math.isfinite(fpr_proxy_if)
        else float("nan")
    )

    meta = stats.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    metrics = stats.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    f1_stats = metrics.get("F1_score", {})
    if not isinstance(f1_stats, dict):
        f1_stats = {}
    fpr_stats = metrics.get("FPR", {})
    if not isinstance(fpr_stats, dict):
        fpr_stats = {}
    alert_stats = metrics.get("Alert_rate_per_1k", {})
    if not isinstance(alert_stats, dict):
        alert_stats = {}

    f1_ci = f1_stats.get("ci_95", [float("nan"), float("nan")])
    if not isinstance(f1_ci, list) or len(f1_ci) != 2:
        f1_ci = [float("nan"), float("nan")]
    fpr_ci = fpr_stats.get("ci_95", [float("nan"), float("nan")])
    if not isinstance(fpr_ci, list) or len(fpr_ci) != 2:
        fpr_ci = [float("nan"), float("nan")]

    correction = str(meta.get("correction_method", "unknown")).upper()
    alpha = _to_float(meta.get("alpha"))
    recommended_test = str(meta.get("recommended_test", "paired test"))

    return (
        "Under a matched operational evaluation, the selected MORL policy achieved "
        f"F1={_fmt(f1_morl)} versus the classical baseline F1={_fmt(f1_if)} "
        f"(ΔF1={_fmt(delta_f1)}). Relative alert volume was {_fmt(fpr_proxy_morl)} vs {_fmt(fpr_proxy_if)} "
        f"alerts per 1k (Δ={_fmt(delta_alerts)}). Paired significance analysis used {recommended_test} "
        f"with bootstrap 95% confidence intervals and {correction} multiple-testing correction at α={_fmt(alpha, 2)}; "
        f"for F1, p_adj={_fmt(_to_float(f1_stats.get('p_adj')), 4)}, effect size dz={_fmt(_to_float(f1_stats.get('effect_size')))}, "
        f"CI=[{_fmt(_to_float(f1_ci[0]))}, {_fmt(_to_float(f1_ci[1]))}]. For FPR, "
        f"p_adj={_fmt(_to_float(fpr_stats.get('p_adj')), 4)}, effect size dz={_fmt(_to_float(fpr_stats.get('effect_size')))}, "
        f"CI=[{_fmt(_to_float(fpr_ci[0]))}, {_fmt(_to_float(fpr_ci[1]))}]. "
        f"Alert-rate significance was p_adj={_fmt(_to_float(alert_stats.get('p_adj')), 4)}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render paper-facing Results paragraph from run outputs"
    )
    parser.add_argument("--row-csv", type=Path, required=True, help="Path to paper_results_row.csv")
    parser.add_argument(
        "--stats-json", type=Path, required=True, help="Path to statistical_analysis.json"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output markdown/text path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = _load_row(args.row_csv)
    stats = _load_json(args.stats_json)
    paragraph = build_paragraph(row, stats)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(paragraph + "\n")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
