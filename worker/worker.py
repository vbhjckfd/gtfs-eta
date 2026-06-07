"""
Cloudflare Python Worker — GTFS-RT TripUpdates passthrough.

on_fetch: serves the pre-computed TripUpdates protobuf from R2.
on_scheduled: fires every 5 minutes (wrangler.toml [triggers]) and dispatches
  the GitHub Actions push-feed workflow via the GitHub API.  The workflow runs
  for ~4 minutes pushing a fresh feed to R2 every 30 s — Cloudflare's cron is
  far more reliable than GitHub's own scheduled triggers on low-activity repos.

Required secret (set via `wrangler secret put GITHUB_TOKEN`):
  GITHUB_TOKEN — a GitHub PAT with the `workflow` scope.
"""

import js
from cloudflare.workers import Response

FEED_KEY = "feed/trip_updates.pb"


async def on_fetch(request, env, ctx=None):
    feed_key = getattr(env, "FEED_KEY", FEED_KEY)
    obj = await env.R2.get(feed_key)
    if obj is None:
        return Response(
            "Feed unavailable — push daemon (scripts/push_feed.py) not running",
            status=503,
        )
    data = bytes(js.Uint8Array.new(await obj.arrayBuffer()))
    return Response(data, headers={
        "content-type":  "application/x-protobuf",
        "cache-control": "public, max-age=30",
    })


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
