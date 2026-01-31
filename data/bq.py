from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover - optional dependency
    bigquery = None


@dataclass(frozen=True)
class BQTableRef:
    project: str
    dataset: str
    table: str

    @property
    def fqtn(self) -> str:
        return f"{self.project}.{self.dataset}.{self.table}"


def bq_client() -> bigquery.Client:
    if bigquery is None:  # pragma: no cover - optional dependency
        raise ImportError("google-cloud-bigquery is required for BigQuery sources")
    return bigquery.Client()


def fetch_rows_timewindow(
    client: bigquery.Client,
    table: BQTableRef,
    ts_col: str,
    start_iso: str,
    end_iso: str,
    limit: Optional[int] = None,
    extra_where: str = "",
) -> Iterable[Mapping[str, Any]]:
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    extra = f"AND ({extra_where})" if extra_where else ""

    query = f"""
    SELECT *
    FROM `{table.fqtn}`
    WHERE {ts_col} >= @start_ts AND {ts_col} < @end_ts
    {extra}
    ORDER BY {ts_col}
    {limit_sql}
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", start_iso),
            bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", end_iso),
        ]
    )

    results = client.query(query, job_config=job_config).result(page_size=10_000)
    for row in results:
        yield dict(row)


def fetch_rows_all(
    client: bigquery.Client,
    table: BQTableRef,
    limit: Optional[int] = None,
    order_by: Optional[str] = None,
) -> Iterable[Mapping[str, Any]]:
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    order_sql = f"ORDER BY {order_by}" if order_by else ""

    query = f"""
    SELECT *
    FROM `{table.fqtn}`
    {order_sql}
    {limit_sql}
    """
    results = client.query(query).result(page_size=10_000)
    for row in results:
        yield dict(row)
