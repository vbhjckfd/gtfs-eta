"""
New Relic custom-event reporting over the raw Event API.

The daemon's half of worker/newrelic.js, with the same no-SDK approach: the
push job lives ~5.5 min, too short for the Python APM agent's connect and
60 s harvest cycle to be worth carrying, and one POST per event is all the
error reporting needs.

Events land in NRDB as their own eventType and are queryable by NRQL, e.g.
  SELECT count(*) FROM GtfsEtaDaemonError FACET errorClass SINCE 1 day ago
  SELECT * FROM GtfsEtaDaemonError WHERE level = 'error' SINCE 6 hours ago
  SELECT sum(pushes), sum(failedPushes) FROM GtfsEtaDaemonRun TIMESERIES SINCE 1 day ago

Optional env:
  NEW_RELIC_LICENSE_KEY — the 40-char ingest licence key (starts eu01xx for
  this account), NOT an NRAK-... user API key.  Without it reporting is off,
  so local runs and tests send nothing.
  NEW_RELIC_ACCOUNT_ID  — defaults to the account the worker reports to.
"""

from __future__ import annotations

import os
import traceback

import requests

# The account is in New Relic's EU region, so ingest must go to the EU
# collector — the US host silently rejects EU licence keys.
EVENT_API_HOST = "https://insights-collector.eu01.nr-data.net"

ACCOUNT_ID = "3352365"

# NRDB caps a string attribute at 4096 bytes; longer values are rejected.
_MAX_ATTR_CHARS = 4000


def record_event(event_type: str, attributes: dict, *, service: str,
                 commit: str | None = None) -> bool:
    """POST one custom event; True when New Relic accepted it.

    Never raises — reporting must not break the feed.
    """
    key = os.environ.get("NEW_RELIC_LICENSE_KEY", "")
    if not key:
        return False
    account_id = os.environ.get("NEW_RELIC_ACCOUNT_ID", ACCOUNT_ID)
    event = {"eventType": event_type, "service": service, "commit": commit}
    for k, v in attributes.items():
        event[k] = v[:_MAX_ATTR_CHARS] if isinstance(v, str) else v
    try:
        resp = requests.post(
            f"{EVENT_API_HOST}/v1/accounts/{account_id}/events",
            headers={"Api-Key": key, "Content-Type": "application/json"},
            json=[event],
            timeout=5,
        )
        if resp.status_code >= 300:
            print(f"[newrelic] event POST failed: HTTP {resp.status_code}", flush=True)
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — reporting must never break the caller
        print(f"[newrelic] failed to record {event_type}: {exc!r}", flush=True)
        return False


def exception_attributes(exc: BaseException) -> dict:
    """errorClass / message / the traceback's last frames, for an error event."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "errorClass": type(exc).__name__,
        "message": str(exc),
        # Keep the tail: the innermost frames are where the failure is.
        "traceback": tb[-_MAX_ATTR_CHARS:],
    }
