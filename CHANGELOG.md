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
