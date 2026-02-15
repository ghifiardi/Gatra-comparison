from typing import Callable, TypeVar, cast

import pandas as pd
import streamlit as st
from google.auth import default as google_auth_default
from google.cloud import bigquery
from google.oauth2 import service_account

F = TypeVar("F", bound=Callable[..., object])

st.set_page_config(page_title="ADA Top-K Queue", layout="wide")

PROJECT = "gatra-prd-c335"
DATASET = "gatra_database"
SAFE_VIEW = f"`{PROJECT}.{DATASET}.vw_ada_queue_streamlit_safe`"

# Auth:
# - Streamlit Cloud: provide [gcp_service_account] in secrets
# - Cloud Run/local ADC: fall back to default credentials
if "gcp_service_account" in st.secrets:
    from_sa_info = cast(
        Callable[[object], service_account.Credentials],
        service_account.Credentials.from_service_account_info,
    )
    creds = from_sa_info(st.secrets["gcp_service_account"])
    client = bigquery.Client(credentials=creds, project=creds.project_id)
else:
    adc_creds, adc_project = google_auth_default()
    client = bigquery.Client(
        credentials=adc_creds,
        project=adc_project or PROJECT,
    )

st.title("ADA Top-200 Queue (per snapshot_dt)")


@cast(Callable[[F], F], st.cache_data(ttl=300))
def load_dates() -> pd.DataFrame:
    query = f"SELECT DISTINCT snapshot_dt FROM {SAFE_VIEW} ORDER BY snapshot_dt DESC"
    return client.query(query).to_dataframe(create_bqstorage_client=False)


@cast(Callable[[F], F], st.cache_data(ttl=300))
def load_queue(snapshot_dt: str) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {SAFE_VIEW}
    WHERE snapshot_dt = DATE(@dt)
    ORDER BY rk ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("dt", "STRING", snapshot_dt)]
    )
    return client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=False)


dates_df = load_dates()
if dates_df.empty:
    st.warning("No data found in the safe view. Check the view/table.")
    st.stop()

snapshot_dt = st.selectbox("snapshot_dt", dates_df["snapshot_dt"].astype(str).tolist())
df = load_queue(snapshot_dt)

left, right = st.columns([2, 1])
with left:
    st.subheader(f"Queue for {snapshot_dt}")
    st.dataframe(df, use_container_width=True, height=600)

with right:
    st.subheader("Quick stats")
    st.metric("Rows", len(df))
    if "prob_y0" in df.columns:
        st.metric("Max prob_y0", float(df["prob_y0"].max()))
        st.metric("Min prob_y0", float(df["prob_y0"].min()))
        st.metric("Avg prob_y0", float(df["prob_y0"].mean()))

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"ada_queue_{snapshot_dt}.csv",
        mime="text/csv",
    )
