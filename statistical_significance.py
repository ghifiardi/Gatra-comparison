from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _as_float_array(values: Sequence[float]) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _safe_float(value: float) -> float:
    if math.isfinite(value):
        return float(value)
    return float("nan")


def _mean_and_ci(
    diffs: FloatArray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    if diffs.size == 0:
        nan = float("nan")
        return nan, nan, nan
    mean_diff = float(np.mean(diffs))
    samples = np.empty(n_bootstrap, dtype=np.float64)
    n = int(diffs.size)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(diffs[idx]))
    low = float(np.percentile(samples, 2.5))
    high = float(np.percentile(samples, 97.5))
    return mean_diff, low, high


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    out = [1.0] * n
    prev = 1.0
    for rank_rev, idx in enumerate(reversed(order), start=1):
        rank = n - rank_rev + 1
        p = p_values[idx]
        adj = min(prev, p * n / rank)
        out[idx] = float(min(1.0, adj))
        prev = adj
    return out


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    raw = [0.0] * n
    for i, idx in enumerate(order):
        raw[idx] = min(1.0, p_values[idx] * (n - i))
    out = raw[:]
    max_so_far = 0.0
    for idx in order:
        max_so_far = max(max_so_far, out[idx])
        out[idx] = max_so_far
    return [float(min(1.0, v)) for v in out]


def _paired_effect_size_dz(diffs: FloatArray) -> float:
    if diffs.size < 2:
        return float("nan")
    std = float(np.std(diffs, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(diffs) / std)


def _paired_ttest_power(effect_size_dz: float, n: int, alpha: float) -> float:
    if n < 2 or not math.isfinite(effect_size_dz):
        return float("nan")
    df = n - 1
    ncp = effect_size_dz * math.sqrt(n)
    t_crit = float(stats.t.ppf(1 - alpha / 2.0, df))
    lower = float(stats.nct.cdf(-t_crit, df, ncp))
    upper = float(stats.nct.cdf(t_crit, df, ncp))
    power = 1.0 - (upper - lower)
    return float(max(0.0, min(1.0, power)))


def _metric_from_confusion(y_true: NDArray[np.int_], y_pred: NDArray[np.int_], key: str) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    if key == "F1_score":
        denom = 2.0 * tp + fp + fn
        return float((2.0 * tp / denom) if denom > 0 else 0.0)
    if key == "FPR":
        denom = fp + tn
        return float((fp / denom) if denom > 0 else 0.0)
    if key == "Alert_rate_per_1k":
        return float(1000.0 * np.mean(y_pred == 1))
    raise KeyError(key)


def bootstrap_metric_samples(
    y_true: NDArray[np.int_],
    y_score: FloatArray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
    ttt_minutes: FloatArray | None = None,
) -> dict[str, FloatArray]:
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have same length")
    if ttt_minutes is not None and ttt_minutes.shape[0] != y_true.shape[0]:
        raise ValueError("ttt_minutes must have same length as y_true")
    mask = y_true != -1
    y = y_true[mask]
    s = y_score[mask]
    ttt = ttt_minutes[mask] if ttt_minutes is not None else None
    if y.size == 0:
        nan_arr = np.full(n_bootstrap, np.nan, dtype=np.float64)
        out = {"F1_score": nan_arr, "FPR": nan_arr, "Alert_rate_per_1k": nan_arr}
        if ttt is not None:
            out["TTT_minutes"] = nan_arr
        return out

    rng = np.random.default_rng(seed)
    f1 = np.empty(n_bootstrap, dtype=np.float64)
    fpr = np.empty(n_bootstrap, dtype=np.float64)
    alerts = np.empty(n_bootstrap, dtype=np.float64)
    ttt_out = np.empty(n_bootstrap, dtype=np.float64) if ttt is not None else None
    n = int(y.size)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        sb = s[idx]
        pred = (sb >= threshold).astype(np.int_)
        f1[i] = _metric_from_confusion(yb, pred, "F1_score")
        fpr[i] = _metric_from_confusion(yb, pred, "FPR")
        alerts[i] = _metric_from_confusion(yb, pred, "Alert_rate_per_1k")
        if ttt_out is not None and ttt is not None:
            ttt_boot = ttt[idx]
            alerted = pred == 1
            ttt_out[i] = float(np.mean(ttt_boot[alerted])) if np.any(alerted) else 0.0
    out = {"F1_score": f1, "FPR": fpr, "Alert_rate_per_1k": alerts}
    if ttt_out is not None:
        out["TTT_minutes"] = ttt_out
    return out


def extract_metric_samples(payload: Mapping[str, Any]) -> dict[str, FloatArray]:
    out: dict[str, FloatArray] = {}

    metric_samples = payload.get("metric_samples")
    if isinstance(metric_samples, Mapping):
        for key, value in metric_samples.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                vals = [float(v) for v in value if isinstance(v, (int, float))]
                if vals:
                    out[str(key)] = _as_float_array(vals)

    samples = payload.get("samples")
    if isinstance(samples, Sequence) and samples and all(isinstance(s, Mapping) for s in samples):
        keys: set[str] = set()
        for row in samples:
            keys.update(str(k) for k in row.keys())
        for key in keys:
            vals: list[float] = []
            for row in samples:
                if not isinstance(row, Mapping):
                    continue
                v = row.get(key)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if vals:
                out[key] = _as_float_array(vals)

    # If explicit sample payloads exist, prefer them over scalar summary fields.
    if out:
        return out

    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        for key, value in metrics.items():
            skey = str(key)
            if skey in out:
                continue
            if isinstance(value, (int, float)):
                out[skey] = _as_float_array([float(value)])
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                vals = [float(v) for v in value if isinstance(v, (int, float))]
                if vals:
                    out[skey] = _as_float_array(vals)

    for key, value in payload.items():
        skey = str(key)
        if skey in out:
            continue
        if isinstance(value, (int, float)):
            out[skey] = _as_float_array([float(value)])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            vals = [float(v) for v in value if isinstance(v, (int, float))]
            if vals:
                out[skey] = _as_float_array(vals)

    return out


class GATRAStatisticalAnalyzer:
    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        correction_method: str = "bh",
        seed: int = 42,
    ) -> None:
        if correction_method not in {"bh", "holm"}:
            raise ValueError("correction_method must be 'bh' or 'holm'")
        self.alpha = float(alpha)
        self.n_bootstrap = int(n_bootstrap)
        self.correction_method = correction_method
        self.seed = int(seed)

    def full_comparison(
        self,
        morl_scores: Mapping[str, FloatArray],
        classical_scores: Mapping[str, FloatArray],
    ) -> dict[str, Any]:
        metric_names = sorted(set(morl_scores.keys()) & set(classical_scores.keys()))
        comparisons: dict[str, dict[str, Any]] = {}
        p_keys: list[str] = []
        p_values: list[float] = []

        for i, metric in enumerate(metric_names):
            a = np.asarray(morl_scores[metric], dtype=np.float64)
            b = np.asarray(classical_scores[metric], dtype=np.float64)
            n = int(min(a.size, b.size))
            if n == 0:
                comparisons[metric] = {
                    "n": 0,
                    "test_name": "insufficient_n",
                    "p_value": float("nan"),
                    "p_adj": float("nan"),
                    "mean_morl": float("nan"),
                    "mean_classical": float("nan"),
                    "mean_diff": float("nan"),
                    "ci_95": [float("nan"), float("nan")],
                    "effect_size": float("nan"),
                    "power": float("nan"),
                }
                continue

            xa = a[:n]
            xb = b[:n]
            diffs = xa - xb

            mean_diff, ci_low, ci_high = _mean_and_ci(
                diffs, self.n_bootstrap, np.random.default_rng(self.seed + i)
            )
            effect = _paired_effect_size_dz(diffs)
            power = _paired_ttest_power(effect, n=n, alpha=self.alpha)
            mean_morl = float(np.mean(xa))
            mean_classical = float(np.mean(xb))

            if n < 2:
                test_name = "insufficient_n"
                p_value = float("nan")
            elif np.allclose(diffs, 0.0):
                test_name = "wilcoxon"
                p_value = 1.0
            else:
                try:
                    test_name = "wilcoxon"
                    p_value = float(stats.wilcoxon(xa, xb, zero_method="wilcox").pvalue)
                except Exception:
                    test_name = "paired_t"
                    p_value = float(stats.ttest_rel(xa, xb, nan_policy="omit").pvalue)

            comparisons[metric] = {
                "n": n,
                "test_name": test_name,
                "p_value": _safe_float(p_value),
                "p_adj": float("nan"),
                "mean_morl": _safe_float(mean_morl),
                "mean_classical": _safe_float(mean_classical),
                "mean_diff": _safe_float(mean_diff),
                "ci_95": [_safe_float(ci_low), _safe_float(ci_high)],
                "effect_size": _safe_float(effect),
                "power": _safe_float(power),
            }
            if math.isfinite(p_value):
                p_keys.append(metric)
                p_values.append(p_value)

        if p_values:
            if self.correction_method == "holm":
                adjusted = _holm_adjust(p_values)
            else:
                adjusted = _bh_adjust(p_values)
            for key, padj in zip(p_keys, adjusted):
                comparisons[key]["p_adj"] = _safe_float(padj)
                comparisons[key]["significant"] = bool(padj < self.alpha)

        return {
            "metadata": {
                "alpha": self.alpha,
                "n_bootstrap": self.n_bootstrap,
                "correction_method": self.correction_method,
                "recommended_test": "wilcoxon",
            },
            "metrics": comparisons,
        }

    def generate_paper_table(self, results: Mapping[str, Any], output_path: str) -> None:
        metrics = results.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        lines = [
            "\\begin{tabular}{lrrrrrr}",
            "\\hline",
            "Metric & MORL & Classical & $\\Delta$ & 95\\% CI & $p_{adj}$ & Effect \\\\",
            "\\hline",
        ]
        for metric in sorted(metrics.keys()):
            row = metrics[metric]
            if not isinstance(row, Mapping):
                continue
            ci = row.get("ci_95", [float("nan"), float("nan")])
            if isinstance(ci, Sequence) and len(ci) == 2:
                ci_text = f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
            else:
                ci_text = "[nan, nan]"
            lines.append(
                f"{metric} & {float(row.get('mean_morl', float('nan'))):.4f}"
                f" & {float(row.get('mean_classical', float('nan'))):.4f}"
                f" & {float(row.get('mean_diff', float('nan'))):.4f}"
                f" & {ci_text}"
                f" & {float(row.get('p_adj', float('nan'))):.4g}"
                f" & {float(row.get('effect_size', float('nan'))):.4f} \\\\"
            )
        lines.extend(["\\hline", "\\end{tabular}", ""])
        Path(output_path).write_text("\n".join(lines))


def run_analysis(
    morl_results_path: str,
    classical_results_path: str,
    output_path: str,
    alpha: float,
    n_bootstrap: int,
    correction_method: str = "bh",
    table_output: str | None = None,
) -> dict[str, Any]:
    morl_payload = json.loads(Path(morl_results_path).read_text())
    classical_payload = json.loads(Path(classical_results_path).read_text())
    if not isinstance(morl_payload, Mapping):
        raise ValueError("morl-results JSON must be an object")
    if not isinstance(classical_payload, Mapping):
        raise ValueError("classical-results JSON must be an object")

    morl_scores = extract_metric_samples(morl_payload)
    classical_scores = extract_metric_samples(classical_payload)

    analyzer = GATRAStatisticalAnalyzer(
        alpha=alpha, n_bootstrap=n_bootstrap, correction_method=correction_method
    )
    results = analyzer.full_comparison(morl_scores, classical_scores)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    if table_output is None:
        table_output = str(out_path.with_name("table1_statistical.tex"))
    analyzer.generate_paper_table(results, table_output)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="GATRA statistical significance analyzer")
    parser.add_argument("--morl-results", required=True)
    parser.add_argument("--classical-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--correction-method", choices=("bh", "holm"), default="bh")
    parser.add_argument("--table-output", default=None)
    args = parser.parse_args()

    run_analysis(
        morl_results_path=args.morl_results,
        classical_results_path=args.classical_results,
        output_path=args.output,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        correction_method=args.correction_method,
        table_output=args.table_output,
    )


if __name__ == "__main__":
    main()
