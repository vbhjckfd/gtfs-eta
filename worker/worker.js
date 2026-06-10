/**
 * Cloudflare Worker — GTFS-RT TripUpdates passthrough.
 *
 * fetch: serves the pre-computed TripUpdates protobuf from R2.
 *   GET /        — raw protobuf feed (drop-in replacement for upstream)
 *   GET /health  — returns JSON 200/503.  Always requires a fresh feed header
 *                  timestamp (< MAX_FEED_AGE_SEC); additionally, during working
 *                  hours, requires stop 60 to have predicted arrivals.  Overnight
 *                  0 arrivals is healthy (transit isn't running), so the arrivals
 *                  check is gated on working hours to avoid false alarms.
 *
 * scheduled: fires every 5 minutes (wrangler.toml [triggers]) and dispatches
 *   the GitHub Actions push-feed workflow via the GitHub API.  The workflow runs
 *   for ~4 minutes pushing a fresh feed to R2 every 30 s — Cloudflare's cron is
 *   far more reliable than GitHub's own scheduled triggers on low-activity repos.
 *
 * This worker was originally a Python Worker (worker.py, git history).  Pyodide
 * isolates intermittently entered a poisoned state where every request died in
 * ~2 ms before handler code ran ("code had hung and would never generate a
 * response", aggregated as scriptThrewException) — unfixable from Python, hence
 * the JS rewrite.  The /health protobuf inspection needs only the feed header
 * timestamp and per-stop stop_time_update counts, so instead of a protobuf
 * dependency we walk the wire format by hand (see parseFeedStats).
 *
 * Required secret (set via `wrangler secret put GITHUB_TOKEN`):
 *   GITHUB_TOKEN — a GitHub PAT with the `workflow` scope.
 *
 * Optional secret (set via `wrangler secret put SENTRY_DSN`):
 *   SENTRY_DSN — enables exception reporting to Sentry via the envelope HTTP
 *   API (no SDK; see sendSentryEvent).  Every event is tagged
 *   service=gtfs-eta-worker so it's distinguishable in a Sentry project shared
 *   with other services.
 */

const FEED_KEY = "feed/trip_updates.pb";

// This DSN is shared with other services (e.g. timetable-api-node); the
// service tag below keeps the worker's events unmistakable in the issue stream.
const SENTRY_SERVICE_TAG = "gtfs-eta-worker";

// /health is layered:
//   1. The feed must be *fresh* — the push pipeline (Cloudflare cron → GitHub
//      Actions → R2) republishes every ~30 s, 24/7.  A stale header timestamp
//      means the pipeline stalled.  Allows one missed 5-min cron cycle + slack.
//   2. During working hours, stop 60 (busiest in Lviv) must also have at least
//      one predicted arrival — the real end-to-end signal that inference is
//      producing predictions, not just that the pipeline is pushing.
// Overnight, Lviv transit isn't running, so 0 arrivals is correct and healthy;
// the arrivals check is therefore gated on working hours.
const MAX_FEED_AGE_SEC = 10 * 60;

// Stop sign-code 60 (Захисників України) → internal GTFS stop_id 4577.
const HEALTH_CHECK_STOP_ID = "4577";

// Working-hours window for the arrivals check, expressed in UTC so we need no
// tzdata/DST logic in the worker.  Lviv is UTC+3 (summer) / UTC+2 (winter), so
// 05:00–18:00 UTC is ~07:00–21:00 local in *both* states — unambiguously inside
// the service day, when the city's busiest stop always has active trips.  The
// edges fall back to the freshness-only check, avoiding false alarms.
const WORKING_HOURS_UTC_START = 5;
const WORKING_HOURS_UTC_END = 18; // exclusive

const textDecoder = new TextDecoder();

/** Cursor over a protobuf wire-format byte range. */
class Reader {
  constructor(buf, pos = 0, end = buf.length) {
    this.buf = buf;
    this.pos = pos;
    this.end = end;
  }

  done() {
    return this.pos >= this.end;
  }

  varint() {
    let result = 0;
    let shift = 0;
    for (;;) {
      if (this.pos >= this.end) throw new Error("truncated varint");
      const b = this.buf[this.pos++];
      // Drop bits above 2^53 — fields we read (timestamps, lengths, keys)
      // never legitimately exceed Number.MAX_SAFE_INTEGER.
      if (shift < 53) result += (b & 0x7f) * 2 ** shift;
      if ((b & 0x80) === 0) return result;
      shift += 7;
    }
  }

  /** Returns [fieldNumber, wireType]. */
  key() {
    const k = this.varint();
    return [Math.floor(k / 8), k & 7];
  }

  /** Reader over a length-delimited submessage, advancing past it. */
  sub() {
    const len = this.varint();
    const start = this.pos;
    this.pos += len;
    if (this.pos > this.end) throw new Error("truncated submessage");
    return new Reader(this.buf, start, this.pos);
  }

  string() {
    const len = this.varint();
    const start = this.pos;
    this.pos += len;
    if (this.pos > this.end) throw new Error("truncated string");
    return textDecoder.decode(this.buf.subarray(start, this.pos));
  }

  skip(wireType) {
    if (wireType === 0) this.varint();
    else if (wireType === 1) this.pos += 8;
    else if (wireType === 2) {
      // NB: not `this.pos += this.varint()` — `+=` reads the old `this.pos`
      // before varint() advances it, silently losing the length-prefix bytes.
      const len = this.varint();
      this.pos += len;
    } else if (wireType === 5) this.pos += 4;
    else throw new Error(`unsupported wire type ${wireType}`);
    if (this.pos > this.end) throw new Error("truncated field");
  }
}

/**
 * Extract what /health needs from a serialized gtfs_realtime.FeedMessage:
 * the header timestamp, the entity count, and how many stop_time_updates
 * reference targetStopId.  Field numbers from gtfs-realtime.proto:
 *   FeedMessage:    header = 1, entity = 2 (repeated)
 *   FeedHeader:     timestamp = 3
 *   FeedEntity:     trip_update = 3
 *   TripUpdate:     stop_time_update = 2 (repeated)
 *   StopTimeUpdate: stop_id = 4
 */
export function parseFeedStats(buf, targetStopId) {
  let timestamp = 0;
  let entities = 0;
  let arrivals = 0;

  const msg = new Reader(buf);
  while (!msg.done()) {
    const [field, wireType] = msg.key();
    if (field === 1 && wireType === 2) {
      const header = msg.sub();
      while (!header.done()) {
        const [hField, hWire] = header.key();
        if (hField === 3 && hWire === 0) timestamp = header.varint();
        else header.skip(hWire);
      }
    } else if (field === 2 && wireType === 2) {
      entities += 1;
      const entity = msg.sub();
      while (!entity.done()) {
        const [eField, eWire] = entity.key();
        if (eField === 3 && eWire === 2) {
          const tripUpdate = entity.sub();
          while (!tripUpdate.done()) {
            const [tField, tWire] = tripUpdate.key();
            if (tField === 2 && tWire === 2) {
              const stu = tripUpdate.sub();
              while (!stu.done()) {
                const [sField, sWire] = stu.key();
                if (sField === 4 && sWire === 2) {
                  if (stu.string() === targetStopId) arrivals += 1;
                } else stu.skip(sWire);
              }
            } else tripUpdate.skip(tWire);
          }
        } else entity.skip(eWire);
      }
    } else msg.skip(wireType);
  }

  return { timestamp, entities, arrivals };
}

/**
 * POST a single event to Sentry's envelope API.  No-op without SENTRY_DSN.
 * Reporting must never break the worker, so all errors here are swallowed.
 */
async function sendSentryEvent(env, fields) {
  const dsn = env.SENTRY_DSN ?? "";
  if (!dsn) return;
  try {
    // DSN: https://<public_key>@<host>/<project_id>
    const [publicKey, hostPath] = dsn.split("://", 2)[1].split("@", 2);
    const slash = hostPath.lastIndexOf("/");
    const host = hostPath.slice(0, slash);
    const projectId = hostPath.slice(slash + 1);
    const envelopeUrl = `https://${host}/api/${projectId}/envelope/`;

    const eventId = crypto.randomUUID().replaceAll("-", "");
    const event = {
      event_id: eventId,
      timestamp: Date.now() / 1000,
      platform: "javascript",
      level: "error",
      environment: env.SENTRY_ENVIRONMENT ?? "production",
      server_name: SENTRY_SERVICE_TAG,
      tags: { service: SENTRY_SERVICE_TAG },
      ...fields,
    };
    if (env.SENTRY_RELEASE) event.release = env.SENTRY_RELEASE;

    const body = [
      JSON.stringify({ event_id: eventId, sent_at: new Date().toISOString() }),
      JSON.stringify({ type: "event", content_type: "application/json" }),
      JSON.stringify(event),
    ].join("\n");

    const resp = await fetch(envelopeUrl, {
      method: "POST",
      headers: {
        "content-type": "application/x-sentry-envelope",
        "x-sentry-auth":
          `Sentry sentry_version=7, sentry_key=${publicKey}, ` +
          `sentry_client=${SENTRY_SERVICE_TAG}/1.0`,
      },
      body,
    });
    if (resp.status >= 300) {
      console.error(`[sentry] envelope POST failed: HTTP ${resp.status}`);
    }
  } catch (reportExc) {
    console.error(`[sentry] failed to report event: ${reportExc}`);
  }
}

async function reportException(env, exc, where) {
  await sendSentryEvent(env, {
    exception: {
      values: [{ type: exc?.name ?? "Error", value: String(exc?.message ?? exc) }],
    },
    extra: { where, stack: exc?.stack ?? "" },
  });
}

async function reportMessage(env, message, where, level = "error") {
  await sendSentryEvent(env, {
    message: { formatted: message },
    level,
    extra: { where },
  });
}

function jsonResponse(payload, status) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function handleHealth(env) {
  const obj = await env.R2.get(env.FEED_KEY ?? FEED_KEY);
  if (obj === null) {
    return jsonResponse({ status: "error", detail: "feed not in R2" }, 503);
  }

  const data = new Uint8Array(await obj.arrayBuffer());
  const { timestamp, entities, arrivals } = parseFeedStats(data, HEALTH_CHECK_STOP_ID);

  const now = Math.floor(Date.now() / 1000);
  const age = now - timestamp;

  if (timestamp === 0 || age > MAX_FEED_AGE_SEC) {
    return jsonResponse({
      status: "stale",
      feed_timestamp: timestamp,
      age_sec: age,
      max_age_sec: MAX_FEED_AGE_SEC,
      entities,
      detail: "feed not fresh — push pipeline may be stalled",
    }, 503);
  }

  // During working hours the busiest stop must have predicted arrivals;
  // outside them, 0 arrivals is expected (transit isn't running).
  const hour = new Date(now * 1000).getUTCHours();
  const workingHours = hour >= WORKING_HOURS_UTC_START && hour < WORKING_HOURS_UTC_END;

  if (workingHours && arrivals === 0) {
    return jsonResponse({
      status: "degraded",
      feed_timestamp: timestamp,
      age_sec: age,
      entities,
      stop_code: 60,
      stop_id: HEALTH_CHECK_STOP_ID,
      arrivals: 0,
      detail:
        "no predicted arrivals for stop 60 during working hours — " +
        "inference may be stalled",
    }, 503);
  }

  return jsonResponse({
    status: "ok",
    feed_timestamp: timestamp,
    age_sec: age,
    entities,
    stop_code: 60,
    stop_id: HEALTH_CHECK_STOP_ID,
    arrivals,
    working_hours: workingHours,
  }, 200);
}

export default {
  async fetch(request, env, ctx) {
    try {
      const path = new URL(request.url).pathname.replace(/\/+$/, "");
      if (path.endsWith("/health")) return await handleHealth(env);

      const obj = await env.R2.get(env.FEED_KEY ?? FEED_KEY);
      if (obj === null) {
        return new Response(
          "Feed unavailable — push daemon (scripts/push_feed.py) not running",
          { status: 503 },
        );
      }
      return new Response(obj.body, {
        headers: {
          "content-type": "application/x-protobuf",
          "cache-control": "public, max-age=30",
        },
      });
    } catch (exc) {
      console.error(`[fetch] unhandled error: ${exc}\n${exc?.stack ?? ""}`);
      ctx.waitUntil(reportException(env, exc, "fetch"));
      return new Response(`Internal error: ${exc}`, { status: 500 });
    }
  },

  async scheduled(event, env, ctx) {
    const repo = env.GITHUB_REPO ?? "vbhjckfd/gtfs-eta";
    const workflow = env.GITHUB_WORKFLOW ?? "push-feed.yml";
    const ref = env.GITHUB_REF ?? "main";
    const token = env.GITHUB_TOKEN ?? "";

    const url =
      `https://api.github.com/repos/${repo}/actions/workflows/${workflow}/dispatches`;
    let resp;
    try {
      resp = await fetch(url, {
        method: "POST",
        headers: {
          "authorization": `Bearer ${token}`,
          "accept": "application/vnd.github+json",
          "content-type": "application/json",
          "user-agent": "gtfs-eta-worker",
          "x-github-api-version": "2022-11-28",
        },
        body: JSON.stringify({ ref }),
      });
    } catch (exc) {
      console.error(`[scheduled] GitHub dispatch raised: ${exc}`);
      await reportException(env, exc, "scheduled");
      return;
    }

    if (resp.status !== 204 && resp.status !== 200) {
      const text = await resp.text();
      const msg = `GitHub dispatch failed: HTTP ${resp.status} — ${text}`;
      console.error(`[scheduled] ${msg}`);
      // A failed dispatch stalls the whole push pipeline, so surface it.
      await reportMessage(env, msg, "scheduled");
    }
  },
};
