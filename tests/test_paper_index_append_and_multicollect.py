from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_index(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "run_group",
                "condition",
                "seed",
                "backend",
                "runner",
                "data_config",
                "morl_config",
                "meta_config",
                "robustness_config",
                "meta_stability_config",
                "run_dir",
                "status",
                "command",
            ]
        )
        writer.writerows(rows)


def test_paper_matrix_append_mode_does_not_duplicate_headers(tmp_path: Path) -> None:
    index_path = tmp_path / "week1_index.csv"
    subprocess.run(
        [
            "bash",
            "scripts/paper_matrix.sh",
            "--csv",
            "--seeds",
            "42",
            "--dry-run",
            "--index-out",
            str(index_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "bash",
            "scripts/paper_matrix.sh",
            "--bq",
            "--bq-seeds",
            "42",
            "--dry-run",
            "--index-out",
            str(index_path),
            "--append-index",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    lines = index_path.read_text().splitlines()
    assert lines
    assert lines[0].startswith("timestamp,")
    assert sum(1 for line in lines if line.startswith("timestamp,")) == 1

    with index_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 8  # 7 CSV conditions + 1 BQ condition
    assert {row["run_group"] for row in rows} == {"csv", "bq"}


def test_collect_multi_index_and_dedupes(tmp_path: Path) -> None:
    run_dir_a = tmp_path / "reports" / "runs" / "run_a"
    run_dir_b = tmp_path / "reports" / "runs" / "run_b"
    run_dir_a.mkdir(parents=True)
    run_dir_b.mkdir(parents=True)

    index_a = tmp_path / "indices" / "index_a.csv"
    index_b = tmp_path / "indices" / "index_b.csv"
    _write_index(
        index_a,
        [
            [
                "2026-02-10T00:00:00Z",
                "csv",
                "A1_csv_option2_default",
                "42",
                "csv",
                "make",
                "cfg/data_csv.yaml",
                "cfg/morl_norm.yaml",
                "cfg/meta_relaxed.yaml",
                "",
                "",
                str(run_dir_a),
                "ok",
                "make run ...",
            ]
        ],
    )
    _write_index(
        index_b,
        [
            [
                "2026-02-10T00:00:00Z",
                "csv",
                "A1_csv_option2_default",
                "42",
                "csv",
                "make",
                "cfg/data_csv.yaml",
                "cfg/morl_norm.yaml",
                "cfg/meta_relaxed.yaml",
                "",
                "",
                str(run_dir_a),
                "ok",
                "make run ...",
            ],
            [
                "2026-02-10T00:01:00Z",
                "bq",
                "A2_bq_option2_replication",
                "42",
                "bigquery",
                "make",
                "cfg/data_bq.yaml",
                "cfg/morl_norm.yaml",
                "cfg/meta_relaxed.yaml",
                "",
                "",
                str(run_dir_b),
                "ok",
                "make run ...",
            ],
        ],
    )

    out_csv = tmp_path / "paper_week1_results.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_paper_results.py",
            "--index",
            str(index_a),
            "--index",
            str(index_b),
            "--out",
            str(out_csv),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    with out_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["run_id"] for row in rows} == {"run_a", "run_b"}

    out_glob_csv = tmp_path / "paper_week1_results_glob.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_paper_results.py",
            "--index-glob",
            str(tmp_path / "indices" / "index_*.csv"),
            "--out",
            str(out_glob_csv),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    with out_glob_csv.open("r", newline="") as f:
        glob_rows = list(csv.DictReader(f))
    assert len(glob_rows) == 2
