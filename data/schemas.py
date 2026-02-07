from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field

LabelType = Literal["threat", "benign", "unknown"]


class RawEvent(BaseModel):
    event_id: str
    ts: datetime
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    port: Optional[int] = None
    protocol: Optional[str] = None
    duration: Optional[float] = None
    bytes_sent: Optional[float] = None
    bytes_received: Optional[float] = None
    user_id: Optional[str] = None
    host_id: Optional[str] = None
    # Optional traceability fields (useful for CSV / raw exports)
    row_key: Optional[str] = None
    event_timestamp_epoch_s: Optional[int] = None
    session_id: Optional[str] = None
    user: Optional[str] = None
    action: Optional[str] = None
    page: Optional[str] = None
    details: Optional[str] = None


class Label(BaseModel):
    event_id: str
    label: LabelType = Field(default="unknown")
    severity: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "unknown"
