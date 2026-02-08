from __future__ import annotations

import json
import os
from typing import Any


def write_meta_selection_artifacts(
    out_dir: str,
    selection: dict[str, Any],
    selected_test: dict[str, Any],
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    selection_json_path = os.path.join(out_dir, "meta_selection.json")
    with open(selection_json_path, "w") as f:
        json.dump(selection, f, indent=2)

    selected_test_json_path = os.path.join(out_dir, "morl_selected_test.json")
    with open(selected_test_json_path, "w") as f:
        json.dump(selected_test, f, indent=2)

    lines = [
        "# Meta-Controller Selection",
        "",
        f"- Selected weight: `{selection.get('selected_weight')}`",
        f"- Method: `{selection.get('method', {}).get('name')}`",
        f"- Candidate count: {selection.get('candidate_count')}",
        f"- Feasible count: {selection.get('feasible_count')}",
        "",
        "## Constraints",
    ]
    constraints = selection.get("constraints", {})
    if isinstance(constraints, dict):
        for key, value in constraints.items():
            lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Selected VAL metrics",
        ]
    )
    metrics = selection.get("selected_val_metrics", {})
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Selected TEST metrics",
        ]
    )
    test_metrics = selected_test.get("metrics", {})
    if isinstance(test_metrics, dict):
        for key, value in test_metrics.items():
            lines.append(f"- {key}: {value}")

    selection_md_path = os.path.join(out_dir, "meta_selection.md")
    with open(selection_md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    selected_lines = [
        "# Selected Weight Test Evaluation",
        "",
        f"- Weight: `{selected_test.get('w')}`",
        "",
        "## Metrics",
    ]
    if isinstance(test_metrics, dict):
        for key, value in test_metrics.items():
            selected_lines.append(f"- {key}: {value}")

    selected_md_path = os.path.join(out_dir, "morl_selected_test.md")
    with open(selected_md_path, "w") as f:
        f.write("\n".join(selected_lines) + "\n")

    return {
        "meta_selection_json": selection_json_path,
        "meta_selection_md": selection_md_path,
        "selected_test_json": selected_test_json_path,
        "selected_test_md": selected_md_path,
    }
