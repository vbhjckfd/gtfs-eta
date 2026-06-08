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
"""

import json

import js
from cloudflare.workers import Response

FEED_KEY = "feed/trip_updates.pb"

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


async def _get_feed_data(env) -> bytes | None:
    feed_key = getattr(env, "FEED_KEY", FEED_KEY)
    obj = await env.R2.get(feed_key)
    if obj is None:
        return None
    return bytes(js.Uint8Array.new(await obj.arrayBuffer()))


async def on_fetch(request, env, ctx=None):
    # Guard the whole handler: on a cold isolate an awaited R2 promise can
    # reject in a way that surfaces via Pyodide's event-loop exception handler
    # rather than propagating, which orphans the response promise and the
    # runtime reports "code had hung and would never generate a response"
    # (a 500 with no body).  Catching here turns that into a clean 500 *and*
    # logs the real traceback so the failure is diagnosable.
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
    except Exception as exc:
        import traceback
        print(f"[on_fetch] unhandled error: {exc!r}\n{traceback.format_exc()}")
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

    if resp.status not in (204, 200):
        text = await resp.text()
        print(f"[scheduled] GitHub dispatch failed: HTTP {resp.status} — {text}")
