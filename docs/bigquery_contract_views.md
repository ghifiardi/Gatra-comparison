# BigQuery contract views (recommended)

These views stabilize the schema expected by the v0.2 loader and frozen contract export.
Point `configs/data.yaml` to these views to avoid frequent mapping updates when raw tables evolve.

## 1) Events view: v_events_contract

```sql
CREATE OR REPLACE VIEW `chronicle-dev-2be9.gatra_database.v_events_contract` AS
SELECT
  CAST(event_id AS STRING) AS event_id,
  TIMESTAMP(ts) AS ts,
  CAST(src_ip AS STRING) AS src_ip,
  CAST(dst_ip AS STRING) AS dst_ip,
  CAST(port AS INT64) AS port,
  LOWER(CAST(protocol AS STRING)) AS protocol,
  CAST(duration AS FLOAT64) AS duration,
  CAST(bytes_sent AS FLOAT64) AS bytes_sent,
  CAST(bytes_received AS FLOAT64) AS bytes_received,
  CAST(user_id AS STRING) AS user_id,
  CAST(host_id AS STRING) AS host_id
FROM `chronicle-dev-2be9.gatra_database.activity_logs`;
```

## 2) Labels view: v_labels_contract

```sql
CREATE OR REPLACE VIEW `chronicle-dev-2be9.gatra_database.v_labels_contract` AS
SELECT
  CAST(event_id AS STRING) AS event_id,
  CASE
    WHEN LOWER(CAST(label AS STRING)) IN ("threat", "benign", "unknown")
      THEN LOWER(CAST(label AS STRING))
    ELSE "unknown"
  END AS label,
  CAST(severity AS FLOAT64) AS severity,
  CAST(source AS STRING) AS source
FROM `chronicle-dev-2be9.gatra_database.ada_feedback`;
```

## 3) Config update

Set the config to point to the views:

- `bq_events_table: "v_events_contract"`
- `bq_labels_table: "v_labels_contract"`
