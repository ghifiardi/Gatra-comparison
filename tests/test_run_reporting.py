from __future__ import annotations

from runs.reporting import build_run_manifest, render_summary_md


def test_build_run_manifest_minimal() -> None:
    manifest = build_run_manifest(
        run_id="20260131T141500Z",
        created_at="2026-01-31T14:15:00Z",
        git_commit="deadbeef",
        config_hashes={"data": "abc", "iforest": "def"},
        config_snapshot={
            "data": "reports/runs/20260131T141500Z/config/data.yaml",
            "iforest": "reports/runs/20260131T141500Z/config/iforest.yaml",
            "ppo": "reports/runs/20260131T141500Z/config/ppo.yaml",
            "eval": "reports/runs/20260131T141500Z/config/eval.yaml",
        },
        data_cfg={
            "dataset": {
                "source": "bigquery",
                "bq_project": "proj",
                "bq_dataset": "ds",
                "bq_events_table": "events",
                "path": "/tmp/events.parquet",
            },
            "labels": {
                "bq_labels_table": "labels",
                "path": "/tmp/labels.parquet",
            },
        },
        schema_hash="schema123",
        poetry_lock_hash="lock123",
        contract_id="20260131T141500Z",
        contract_meta={"counts": {"train": 1}, "label_pos_rate": {"train": 0.1}},
        mode="quick",
        seeds={"python": 42, "numpy": 42, "torch": 42, "iforest": 42},
    )
    assert manifest["run_id"] == "20260131T141500Z"
    assert manifest["schema_hash"] == "schema123"
    assert manifest["config_hashes"]["data"] == "abc"
    assert manifest["data_source"]["bq_events_table"] == "events"
    assert manifest["poetry_lock_sha256"] == "lock123"
    assert manifest["mode"] == "quick"
    assert manifest["config_snapshot"]["data"].endswith("data.yaml")
    assert manifest["seeds"]["torch"] == 42


def test_render_summary_md_contains_metrics() -> None:
    summary = render_summary_md(
        run_id="20260131T141500Z",
        git_commit="deadbeef",
        schema_hash="schema123",
        contract_meta={
            "counts": {"train": 10, "val": 5, "test": 3},
            "label_pos_rate": {"train": 0.1, "val": 0.2, "test": 0.3},
            "splits": {
                "train": {"start": "2025-01-01", "end": "2025-01-02"},
                "val": {"start": "2025-01-02", "end": "2025-01-03"},
                "test": {"start": "2025-01-03", "end": "2025-01-04"},
            },
        },
        metrics={
            "iforest": {"roc_auc": 0.9, "pr_auc": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65},
            "ppo": {"roc_auc": 0.8, "pr_auc": 0.7, "precision": 0.6, "recall": 0.5, "f1": 0.55},
        },
        thresholds={"iforest": 0.8, "ppo": 0.5},
        train_times={"iforest": 1.0, "ppo": 2.0},
        iforest_cfg={"model": {"n_estimators": 100, "contamination": "auto"}},
        ppo_cfg={"train": {"epochs": 2, "batch_size": 32}, "networks": {"hidden_sizes": [64, 32]}},
        contract_dir="reports/runs/20260131T141500Z/contract",
        mode="quick",
    )
    assert "Run ID: 20260131T141500Z" in summary
    assert "## Results" in summary
    assert "| iforest |" in summary
    assert "| ppo |" in summary
