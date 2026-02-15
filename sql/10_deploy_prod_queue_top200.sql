-- Deploy Top-200 queue (rank by anomaly probability prob_y0 DESC)
DROP TABLE IF EXISTS `gatra-prd-c335.gatra_database.ada_predictions_v4_prod_queue`;

CREATE TABLE `gatra-prd-c335.gatra_database.ada_predictions_v4_prod_queue`
PARTITION BY snapshot_dt
CLUSTER BY rk, alarm_key AS
WITH ranked AS (
  SELECT
    alarm_key,
    snapshot_dt,
    scored_at,
    predicted_y,
    prob_y0,
    prob_y1,
    threshold_p99_non_anom,
    predicted_anomaly_y0,
    ROW_NUMBER() OVER (
      PARTITION BY snapshot_dt
      ORDER BY prob_y0 DESC, scored_at DESC
    ) AS rk
  FROM `gatra-prd-c335.gatra_database.ada_predictions_v4_test_scored`
)
SELECT *
FROM ranked
WHERE rk <= 200;
