"""
Cloudflare Python Worker — GTFS-RT TripUpdates passthrough.

Serves the pre-computed TripUpdates protobuf written to R2 by the external
push daemon (scripts/push_feed.py).  No inference happens here — the Worker
just does one R2 read per request, which costs ~0 ms of CPU and comfortably
fits within Cloudflare's free-plan 10 ms CPU budget.

The push daemon keeps the blob fresh (default: every 30 s).  The
cache-control header lets Cloudflare's edge cache serve most requests from
memory, so R2 is only hit once per cache TTL per edge location.
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
