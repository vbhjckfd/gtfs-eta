/**
 * Workers Analytics Engine reporting — the Cloudflare-native twin of
 * newrelic.js.
 *
 * writeDataPoint is synchronous and fire-and-forget: no fetch, no secret, no
 * latency, and it is free on the Workers Free plan.  Points are queryable over
 * the SQL API (or Grafana) as events in the `gtfs_eta_worker` dataset.
 *
 * Analytics Engine columns are positional (blob1..20, double1..20), so each
 * point kind's layout is fixed here and documented in the README.  index1 is
 * the point kind; every point also carries this worker's commit as blob1.
 * Doubles cannot be null, so an unknown number is written as -1 (none of the
 * fields below is legitimately negative, except feedSkewSec — read it only
 * where vehiclesIn >= 0, since both come from the same feed metadata).
 *
 * Without the METRICS binding (e.g. `wrangler dev` without it) this is a no-op.
 */

function num(v) {
  if (typeof v === "boolean") return v ? 1 : 0;
  return typeof v === "number" && Number.isFinite(v) ? v : -1;
}

function writePoint(env, kind, blobs, doubles) {
  if (!env.METRICS) return;
  try {
    env.METRICS.writeDataPoint({
      indexes: [kind],
      blobs: [env.GIT_COMMIT ?? "unknown", ...blobs.map((b) => String(b ?? ""))],
      doubles: doubles.map(num),
    });
  } catch (exc) {
    console.error(`[analytics] failed to write ${kind} point: ${exc}`);
  }
}

/**
 * index1 = "health"
 *   blob2 status   blob3 feedCommit
 *   double1 httpStatus  double2 ageSec  double3 entities  double4 arrivals
 *   double5 workingHours (0/1)  double6 vehiclesIn  double7 vehiclesStale
 *   double8 feedSkewSec
 */
export function healthPoint(env, httpStatus, body) {
  writePoint(env, "health", [body.status ?? "unknown", body.feed_commit], [
    httpStatus,
    body.age_sec,
    body.entities,
    body.arrivals,
    body.working_hours,
    body.vehicles_in,
    body.vehicles_stale,
    body.feed_skew_sec,
  ]);
}

/**
 * index1 = "cron"
 *   blob2 cron expression   blob3 workflow
 *   double1 dispatched (0/1)  double2 archived (0/1, -1 = not attempted)
 *   double3 durationMs
 */
export function cronPoint(env, { cron, workflow, dispatched, archived, durationMs }) {
  writePoint(env, "cron", [cron, workflow], [dispatched, archived, durationMs]);
}

/**
 * index1 = "watchdog"
 *   blob2 action: ok | cooldown | dispatched | dispatch_failed | error
 *   double1 feed ageSec (-1 = feed missing)  double2 dispatched (0/1)
 */
export function watchdogPoint(env, { action, ageSec, dispatched }) {
  writePoint(env, "watchdog", [action], [ageSec, dispatched]);
}
