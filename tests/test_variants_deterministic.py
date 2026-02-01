from __future__ import annotations

import numpy as np

from evaluation.variants import apply_missingness, apply_noise


def test_missingness_deterministic_same_seed() -> None:
    X = np.arange(200, dtype=np.float32).reshape(20, 10)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out1 = apply_missingness(X, rate=0.1, strategy="mcar", fill="zero", rng=rng1)
    out2 = apply_missingness(X, rate=0.1, strategy="mcar", fill="zero", rng=rng2)
    assert np.array_equal(out1, out2)


def test_missingness_different_seed_changes_mask() -> None:
    X = np.arange(200, dtype=np.float32).reshape(20, 10)
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    out1 = apply_missingness(X, rate=0.2, strategy="mcar", fill="zero", rng=rng1)
    out2 = apply_missingness(X, rate=0.2, strategy="mcar", fill="zero", rng=rng2)
    assert not np.array_equal(out1, out2)


def test_noise_deterministic_same_seed() -> None:
    X = np.ones((16, 8), dtype=np.float32)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    out1 = apply_noise(X, sigma=0.05, distribution="gaussian", clamp=True, rng=rng1)
    out2 = apply_noise(X, sigma=0.05, distribution="gaussian", clamp=True, rng=rng2)
    assert np.allclose(out1, out2)
