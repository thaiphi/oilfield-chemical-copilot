from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


def build_trace_event(event_type: str, payload: dict[str, Any]) -> TraceEvent:
    """Build a trace event for future database ingestion."""
    # TODO: Persist events through dlt or direct PostgreSQL writes.
    return TraceEvent(event_type=event_type, payload=payload, created_at=datetime.now(timezone.utc))
