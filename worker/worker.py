"""
Cloudflare Python Worker — GTFS-RT TripUpdates passthrough.

on_fetch: serves the pre-computed TripUpdates protobuf from R2.
  GET /        — raw protobuf feed (drop-in replacement for upstream)
  GET /health  — checks that stop 60 (Захисників України, internal id 4577)
                 has at least one predicted arrival; returns JSON 200/503.

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

# Stop sign-code 60 (Захисників України) — highest-traffic stop in Lviv.
# Internal GTFS stop_id as it appears in stop_time_update.stop_id.
HEALTH_CHECK_STOP_ID = "4577"


async def _get_feed_data(env) -> bytes | None:
    feed_key = getattr(env, "FEED_KEY", FEED_KEY)
    obj = await env.R2.get(feed_key)
    if obj is None:
        return None
    return bytes(js.Uint8Array.new(await obj.arrayBuffer()))


async def on_fetch(request, env, ctx=None):
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


async def _handle_health(env):
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

    count = sum(
        1
        for e in feed.entity
        for stu in e.trip_update.stop_time_update
        if stu.stop_id == HEALTH_CHECK_STOP_ID
    )

    if count == 0:
        return Response(
            json.dumps({
                "status": "degraded",
                "stop_code": 60,
                "stop_id": HEALTH_CHECK_STOP_ID,
                "arrivals": 0,
                "detail": "no predicted arrivals for stop 60 — inference may be stalled",
            }),
            status=503,
            headers={"content-type": "application/json"},
        )

    return Response(
        json.dumps({
            "status": "ok",
            "stop_code": 60,
            "stop_id": HEALTH_CHECK_STOP_ID,
            "arrivals": count,
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
