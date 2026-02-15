-- Deploy safe Streamlit view (hashed alarm key, no raw identifier exposure).
CREATE OR REPLACE VIEW `gatra-prd-c335.gatra_database.vw_ada_queue_streamlit_safe` AS
SELECT
  snapshot_dt,
  rk,
  TO_HEX(SHA256(alarm_key)) AS alarm_key_hash,
  prob_y0,
  prob_y1,
  predicted_y,
  scored_at
FROM `gatra-prd-c335.gatra_database.ada_predictions_v4_prod_queue`;
