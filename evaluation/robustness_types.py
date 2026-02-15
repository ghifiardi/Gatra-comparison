from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RobustnessVariantResult:
    variant: str
    kind: str
    meta: dict[str, Any]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RobustnessSuiteResult:
    seed: int
    contract_dir: str
    variants: list[RobustnessVariantResult]
