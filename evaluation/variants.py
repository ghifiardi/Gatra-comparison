from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


FloatArr = NDArray[np.float32]
IntArr = NDArray[np.int8]


@dataclass(frozen=True)
class VariantBatch:
    name: str
    X7: FloatArr
    X128: FloatArr
    y: IntArr
    meta: dict[str, Any]


def _ensure_float32(x: NDArray[np.floating[Any]]) -> FloatArr:
    return np.asarray(x, dtype=np.float32)


def _ensure_int8(y: NDArray[np.integer[Any]]) -> IntArr:
    return np.asarray(y, dtype=np.int8)


def apply_time_slice(
    X7: FloatArr,
    X128: FloatArr,
    y: IntArr,
    frac_start: float,
    frac_end: float,
    label: str,
) -> VariantBatch:
    n = int(y.shape[0])
    i0 = int(max(0, min(n, round(frac_start * n))))
    i1 = int(max(0, min(n, round(frac_end * n))))
    if i1 <= i0:
        i1 = min(n, i0 + 1)

    return VariantBatch(
        name=f"time_slice:{label}",
        X7=_ensure_float32(X7[i0:i1]),
        X128=_ensure_float32(X128[i0:i1]),
        y=_ensure_int8(y[i0:i1]),
        meta={"slice": {"label": label, "frac_start": frac_start, "frac_end": frac_end}},
    )


def apply_missingness(
    X: FloatArr,
    rate: float,
    strategy: Literal["mcar", "topk"],
    fill: Literal["zero", "mean", "median"],
    rng: np.random.Generator,
) -> FloatArr:
    Xo = X.copy()
    n, d = Xo.shape
    if rate <= 0:
        return Xo
    k = int(round(rate * n * d))
    k = max(0, min(k, n * d))

    if strategy == "mcar":
        idx = rng.choice(n * d, size=k, replace=False)
        rows = idx // d
        cols = idx % d
    else:
        col_var = np.var(Xo, axis=0)
        top_cols = np.argsort(col_var)[::-1][: max(1, d // 10)]
        flat_idx = rng.choice(n * top_cols.shape[0], size=k, replace=True)
        rows = flat_idx // top_cols.shape[0]
        cols = top_cols[flat_idx % top_cols.shape[0]]

    if fill == "zero":
        Xo[rows, cols] = 0.0
    elif fill == "mean":
        mu = np.mean(Xo, axis=0)
        Xo[rows, cols] = mu[cols]
    else:
        med = np.median(Xo, axis=0)
        Xo[rows, cols] = med[cols]

    return Xo


def apply_noise(
    X: FloatArr,
    sigma: float,
    distribution: Literal["gaussian", "laplace"],
    clamp: bool,
    rng: np.random.Generator,
) -> FloatArr:
    if sigma <= 0:
        return X.copy()

    if distribution == "gaussian":
        noise = rng.normal(loc=0.0, scale=sigma, size=X.shape).astype(np.float32)
    else:
        noise = rng.laplace(loc=0.0, scale=sigma, size=X.shape).astype(np.float32)

    Xo = (X + noise).astype(np.float32)
    if clamp:
        Xo = np.clip(Xo, -10.0, 10.0).astype(np.float32)
    return Xo


def apply_label_delay(
    y: IntArr,
    fraction: float,
    policy: Literal["treat_as_benign", "treat_as_unknown", "drop"],
    rng: np.random.Generator,
) -> tuple[IntArr, dict[str, Any]]:
    """Contract-only delay proxy: degrade a fraction of positives."""
    yo = y.copy()
    pos_idx = np.flatnonzero(yo == 1)
    m = int(round(fraction * pos_idx.shape[0]))
    m = max(0, min(m, pos_idx.shape[0]))
    chosen = rng.choice(pos_idx, size=m, replace=False) if m > 0 else np.array([], dtype=int)

    meta = {"label_delay": {"fraction": fraction, "policy": policy, "affected_pos": int(m)}}

    if policy == "treat_as_benign":
        yo[chosen] = 0
        return yo, meta

    # treat_as_unknown or drop: mark unknown as -1 (filter later)
    yo[chosen] = -1
    return yo, meta
