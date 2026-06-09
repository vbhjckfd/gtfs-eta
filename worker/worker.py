"""
Cloudflare Python Worker — GTFS-RT TripUpdates passthrough.

on_fetch: serves the pre-computed TripUpdates protobuf from R2.
  GET /        — raw protobuf feed (drop-in replacement for upstream)
  GET /health  — returns JSON 200/503.  Always requires a fresh feed header
                 timestamp (< MAX_FEED_AGE_SEC); additionally, during working
                 hours, requires stop 60 to have predicted arrivals.  Overnight
                 0 arrivals is healthy (transit isn't running), so the arrivals
                 check is gated on working hours to avoid false alarms.

on_scheduled: fires every 5 minutes (wrangler.toml [triggers]) and dispatches
  the GitHub Actions push-feed workflow via the GitHub API.  The workflow runs
  for ~4 minutes pushing a fresh feed to R2 every 30 s — Cloudflare's cron is
  far more reliable than GitHub's own scheduled triggers on low-activity repos.

Required secret (set via `wrangler secret put GITHUB_TOKEN`):
  GITHUB_TOKEN — a GitHub PAT with the `workflow` scope.

Optional secret (set via `wrangler secret put SENTRY_DSN`):
  SENTRY_DSN — enables exception reporting to Sentry.  `sentry-sdk` can't run
  under Pyodide, so we POST events directly to Sentry's envelope HTTP API
  (see _report_to_sentry).  Every event is tagged service=gtfs-eta-worker so
  it's distinguishable in a Sentry project shared with other services.
"""

import json

import js
from cloudflare.workers import Response

FEED_KEY = "feed/trip_updates.pb"

# This DSN is shared with other services (e.g. timetable-api-node); the
# service tag below keeps the worker's events unmistakable in the issue stream.
SENTRY_SERVICE_TAG = "gtfs-eta-worker"

# /health is layered:
#   1. The feed must be *fresh* — the push pipeline (Cloudflare cron → GitHub
#      Actions → R2) republishes every ~30 s, 24/7.  A stale header timestamp
#      means the pipeline stalled.  Allows one missed 5-min cron cycle + slack.
#   2. During working hours, stop 60 (busiest in Lviv) must also have at least
#      one predicted arrival — the real end-to-end signal that inference is
#      producing predictions, not just that the pipeline is pushing.
# Overnight, Lviv transit isn't running, so 0 arrivals is correct and healthy;
# the arrivals check is therefore gated on working hours.
MAX_FEED_AGE_SEC = 10 * 60

# Stop sign-code 60 (Захисників України) → internal GTFS stop_id 4577.
HEALTH_CHECK_STOP_ID = "4577"

# Working-hours window for the arrivals check, expressed in UTC so we need no
# tzdata/DST logic in the worker.  Lviv is UTC+3 (summer) / UTC+2 (winter), so
# 05:00–18:00 UTC is ~07:00–21:00 local in *both* states — unambiguously inside
# the service day, when the city's busiest stop always has active trips.  The
# edges fall back to the freshness-only check, avoiding false alarms.
WORKING_HOURS_UTC = range(5, 18)

# Bound every R2 read.  On some (notably cold) isolates the awaited R2 promise's
# *rejection* is delivered to Pyodide's asyncio event-loop exception handler
# instead of raising inside the `await`, so the read hangs forever, on_fetch's
# try/except never sees it, no Response is produced, and the Workers runtime
# kills the request ("code had hung and would never generate a response" — which
# aggregates as scriptThrewException).  Wrapping the read in asyncio.wait_for
# keeps a live timer on the loop (so the runtime doesn't hang-cancel) and lets
# us return a clean 503 on timeout instead.  Normal reads finish in <1.5 s.
#
# NB: do NOT try to surface the diverted rejection by installing an asyncio
# loop exception handler — the loop is reused across requests, and any Python
# state left on it (a handler closure, a create_task'd coroutine) is touched by
# the next request's preparePython without the GIL held, which throws
# "NoGilError: Attempted to use PyProxy when Python GIL not held".  Keep all
# Python work strictly inside an awaited call within the request.
R2_READ_TIMEOUT_SEC = 5


async def _send_sentry_event(env, fields: dict) -> None:
    """POST a single event to Sentry's envelope API. No-op without SENTRY_DSN.

    sentry-sdk doesn't run under Pyodide, so we build the envelope by hand.
    Reporting must never break the worker, so all errors here are swallowed.
    """
    dsn = getattr(env, "SENTRY_DSN", "")
    if not dsn:
        return
    try:
        import time
        import uuid
        from datetime import datetime, timezone

        from pyodide.ffi import to_js

        # DSN: https://<public_key>@<host>/<project_id>
        public_key, host_path = dsn.split("://", 1)[1].split("@", 1)
        host, project_id = host_path.rsplit("/", 1)
        envelope_url = f"https://{host}/api/{project_id}/envelope/"

        event_id = uuid.uuid4().hex
        event = {
            "event_id": event_id,
            "timestamp": time.time(),
            "platform": "python",
            "level": "error",
            "environment": getattr(env, "SENTRY_ENVIRONMENT", "production"),
            "server_name": SENTRY_SERVICE_TAG,
            "tags": {"service": SENTRY_SERVICE_TAG},
            **fields,
        }
        release = getattr(env, "SENTRY_RELEASE", "")
        if release:
            event["release"] = release

        body = "\n".join((
            json.dumps({
                "event_id": event_id,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            }),
            json.dumps({"type": "event", "content_type": "application/json"}),
            json.dumps(event),
        ))

        resp = await js.fetch(envelope_url, to_js({
            "method": "POST",
            "headers": {
                "content-type": "application/x-sentry-envelope",
                "x-sentry-auth": (
                    f"Sentry sentry_version=7, sentry_key={public_key}, "
                    f"sentry_client={SENTRY_SERVICE_TAG}/1.0"
                ),
            },
            "body": body,
        }, dict_converter=js.Object.fromEntries))
        if resp.status >= 300:
            print(f"[sentry] envelope POST failed: HTTP {resp.status}")
    except Exception as report_exc:  # noqa: BLE001 — never break the worker
        print(f"[sentry] failed to report event: {report_exc!r}")


async def _report_exception(env, exc, where: str) -> None:
    import traceback

    await _send_sentry_event(env, {
        "exception": {"values": [{
            "type": type(exc).__name__,
            "value": str(exc),
        }]},
        "extra": {"where": where, "traceback": traceback.format_exc()},
    })


async def _report_message(env, message: str, where: str, level: str = "error") -> None:
    await _send_sentry_event(env, {
        "message": {"formatted": message},
        "level": level,
        "extra": {"where": where},
    })


async def _get_feed_data(env) -> bytes | None:
    import asyncio

    feed_key = getattr(env, "FEED_KEY", FEED_KEY)

    async def _read() -> bytes | None:
        obj = await env.R2.get(feed_key)
        if obj is None:
            return None
        return bytes(js.Uint8Array.new(await obj.arrayBuffer()))

    # See R2_READ_TIMEOUT_SEC: a bare `await` here can hang the whole request.
    return await asyncio.wait_for(_read(), R2_READ_TIMEOUT_SEC)


async def on_fetch(request, env, ctx=None):
    # Guard the whole handler: on a cold isolate an awaited R2 promise can
    # reject in a way that surfaces via Pyodide's event-loop exception handler
    # rather than propagating, which orphans the response promise and the
    # runtime reports "code had hung and would never generate a response"
    # (a 500 with no body).  Catching here turns that into a clean 500 *and*
    # logs the real traceback so the failure is diagnosable.
    import asyncio

    try:
        path = request.url.split("?")[0].rstrip("/")
        if path.endswith("/health"):
            return await _handle_health(env)

        data = await _get_feed_data(env)
        if data is None:
            return Response(
                "Feed unavailable — push daemon (scripts/push_feed.py) not running",
                status=503,
            )
        return Response(data, headers={
            "content-type":  "application/x-protobuf",
            "cache-control": "public, max-age=30",
        })
    except asyncio.TimeoutError as exc:
        # R2 read exceeded R2_READ_TIMEOUT_SEC — almost always the diverted-
        # rejection hang.  Return a clean 503 the caller can retry rather than
        # letting the runtime hang-cancel the request (scriptThrewException).
        print(f"[on_fetch] R2 read timed out after {R2_READ_TIMEOUT_SEC}s")
        await _report_message(
            env, f"R2 read timed out after {R2_READ_TIMEOUT_SEC}s", "on_fetch"
        )
        return Response("Feed read timed out", status=503)
    except Exception as exc:
        import traceback
        print(f"[on_fetch] unhandled error: {exc!r}\n{traceback.format_exc()}")
        await _report_exception(env, exc, "on_fetch")
        return Response(f"Internal error: {exc!r}", status=500)


async def _handle_health(env):
    import time

    data = await _get_feed_data(env)
    if data is None:
        return Response(
            json.dumps({"status": "error", "detail": "feed not in R2"}),
            status=503,
            headers={"content-type": "application/json"},
        )

    from google.transit import gtfs_realtime_pb2
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    now = int(time.time())
    feed_ts = feed.header.timestamp
    age = now - feed_ts
    entities = len(feed.entity)

    if feed_ts == 0 or age > MAX_FEED_AGE_SEC:
        return Response(
            json.dumps({
                "status": "stale",
                "feed_timestamp": feed_ts,
                "age_sec": age,
                "max_age_sec": MAX_FEED_AGE_SEC,
                "entities": entities,
                "detail": "feed not fresh — push pipeline may be stalled",
            }),
            status=503,
            headers={"content-type": "application/json"},
        )

    # During working hours the busiest stop must have predicted arrivals;
    # outside them, 0 arrivals is expected (transit isn't running).
    working_hours = time.gmtime(now).tm_hour in WORKING_HOURS_UTC
    arrivals = sum(
        1
        for e in feed.entity
        for stu in e.trip_update.stop_time_update
        if stu.stop_id == HEALTH_CHECK_STOP_ID
    )

    if working_hours and arrivals == 0:
        return Response(
            json.dumps({
                "status": "degraded",
                "feed_timestamp": feed_ts,
                "age_sec": age,
                "entities": entities,
                "stop_code": 60,
                "stop_id": HEALTH_CHECK_STOP_ID,
                "arrivals": 0,
                "detail": "no predicted arrivals for stop 60 during working "
                          "hours — inference may be stalled",
            }),
            status=503,
            headers={"content-type": "application/json"},
        )

    return Response(
        json.dumps({
            "status": "ok",
            "feed_timestamp": feed_ts,
            "age_sec": age,
            "entities": entities,
            "stop_code": 60,
            "stop_id": HEALTH_CHECK_STOP_ID,
            "arrivals": arrivals,
            "working_hours": working_hours,
        }),
        status=200,
        headers={"content-type": "application/json"},
    )


async def on_scheduled(event, env, ctx):
    import json
    from pyodide.ffi import to_js

    repo     = getattr(env, "GITHUB_REPO",    "vbhjckfd/gtfs-eta")
    workflow = getattr(env, "GITHUB_WORKFLOW", "push-feed.yml")
    ref      = getattr(env, "GITHUB_REF",     "main")
    token    = getattr(env, "GITHUB_TOKEN",   "")

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    try:
        resp = await js.fetch(url, to_js({
            "method": "POST",
            "headers": {
                "authorization":       f"Bearer {token}",
                "accept":              "application/vnd.github+json",
                "content-type":        "application/json",
                "user-agent":          "gtfs-eta-worker",
                "x-github-api-version": "2022-11-28",
            },
            "body": json.dumps({"ref": ref}),
        }, dict_converter=js.Object.fromEntries))
    except Exception as exc:
        print(f"[scheduled] GitHub dispatch raised: {exc!r}")
        await _report_exception(env, exc, "on_scheduled")
        return

    if resp.status not in (204, 200):
        text = await resp.text()
        msg = f"GitHub dispatch failed: HTTP {resp.status} — {text}"
        print(f"[scheduled] {msg}")
        # A failed dispatch stalls the whole push pipeline, so surface it.
        await _report_message(env, msg, "on_scheduled")
