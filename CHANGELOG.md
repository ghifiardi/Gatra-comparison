## v0.11
- BigQuery data loader (`data/bq_loader.py`): load events and labels from
  `gatra-prd-c335.gatra_database.activity_logs` / `ada_feedback`
- `google-cloud-bigquery` added as optional dependency (`poetry install -E bigquery`)
- Lazy import keeps CI and `make test` working without GCP credentials
- Label matching follows CSV loader priority: row_key → alarm_id → event_id
- CI-safe unit tests for row mapping, config validation, and label matching

## v0.10.1
- Operator quickstart tightened for Option 2 defaults (local CSV + BigQuery variants)
- Artifact contract documented for `run_morl_policy_quick` with troubleshooting guidance
- Summary/report guardrail for single-class TEST splits:
  - ROC-AUC/PR-AUC shown as `N/A`
  - explicit note: metric omitted due to one-class TEST labels
- No behavior change to fallback logic; default relaxed meta-controller path remains unchanged

## v0.9
- Contract cache: skip re-export when data config unchanged (keyed by config content hash)
- Resume mode: `--contract-dir-input` reuses a pre-built contract directory
- Cache directory: `reports/contracts_cache/<key>/contract`
- Manifest records `cache_hit` and `contract_source_dir` in contract section
- Tests: test_contract_cache.py (cache miss/hit, copy, key determinism)

## v0.8
- Meta-selection stability suite: replay meta-controller across conditions
- Conditions: robustness variants, label availability delays, policy regimes
- Aggregate metrics: selection_change_rate, avg_weight_L1_distance, constraint_violation_rate, avg_regret
- Artifacts: meta_stability.json, meta_stability.md, meta_stability_table.csv
- CLI integration: `--meta-stability-config` flag with manifest fields
- Make targets: `meta_stability`, `run_meta_stability_quick`
- Deterministic tests: test_meta_stability_metrics.py, test_meta_stability_smoke.py

## v0.1
- Repo scaffold: RL PPO vs Isolation Forest comparison harness
- Toy dataset end-to-end: train_b, train_a, head-to-head eval
- Serving stubs + common inference contract
- Basic tests + lint hooks
