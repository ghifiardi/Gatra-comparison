-- Verify queue table has expected shape and ranking behavior.

-- 1) Total rows
SELECT COUNT(*) AS total_rows
FROM `gatra-prd-c335.gatra_database.ada_predictions_v4_prod_queue`;

-- 2) Per-day row counts
SELECT snapshot_dt, COUNT(*) AS rows_per_day
FROM `gatra-prd-c335.gatra_database.ada_predictions_v4_prod_queue`
GROUP BY snapshot_dt
ORDER BY snapshot_dt DESC
LIMIT 30;

-- 3) Rank range and uniqueness sanity per day
SELECT
  snapshot_dt,
  COUNT(*) AS n_rows,
  MIN(rk) AS min_rk,
  MAX(rk) AS max_rk,
  COUNT(DISTINCT rk) AS distinct_rk
FROM `gatra-prd-c335.gatra_database.ada_predictions_v4_prod_queue`
GROUP BY snapshot_dt
ORDER BY snapshot_dt DESC
LIMIT 30;
