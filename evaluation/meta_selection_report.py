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

    feasibility_payload = selection.get("feasibility", {})
    if not isinstance(feasibility_payload, dict):
        feasibility_payload = {}
    feasibility_json_path = os.path.join(out_dir, "meta_feasibility.json")
    with open(feasibility_json_path, "w") as f:
        json.dump(feasibility_payload, f, indent=2)

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

    lines.extend(["", "## Fallback"])
    lines.append(f"- fallback_used: {feasibility_payload.get('fallback_used', False)}")
    lines.append(f"- fallback_mode: {feasibility_payload.get('fallback_mode', 'none')}")
    lines.append(
        f"- feasible_count_initial: {feasibility_payload.get('feasible_count_initial', 0)}"
    )
    lines.append(
        f"- final_constraints_used: {feasibility_payload.get('final_constraints_used', {})}"
    )
    lines.append(f"- selection_rationale: {feasibility_payload.get('selection_rationale', 'n/a')}")

    relaxation_trace = feasibility_payload.get("relaxation_trace", [])
    if isinstance(relaxation_trace, list) and relaxation_trace:
        lines.extend(
            [
                "",
                "| relax_step | feasible_count | constraints |",
                "| --- | --- | --- |",
            ]
        )
        for row in relaxation_trace:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {row.get('step')} | {row.get('feasible_count')} | {row.get('constraints')} |"
            )

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
        "meta_feasibility_json": feasibility_json_path,
        "meta_selection_md": selection_md_path,
        "selected_test_json": selected_test_json_path,
        "selected_test_md": selected_md_path,
    }
