/**
 * New Relic custom-event reporting over the raw Event API.
 *
 * Same reasoning as sendSentryEvent in worker.js: no SDK.  The New Relic Node
 * APM agent cannot run in a Worker at all — it is CommonJS, spawns background
 * harvest timers and reads Node internals that do not exist in a V8 isolate —
 * so this posts custom events directly over HTTP.
 *
 * Events land in NRDB as their own eventType and are queryable by NRQL, e.g.
 *   SELECT count(*) FROM GtfsEtaWorkerHealth FACET status SINCE 6 hours ago
 *   SELECT max(ageSec) FROM GtfsEtaWorkerHealth TIMESERIES SINCE 1 day ago
 *
 * Optional secret (set via `wrangler secret put NEW_RELIC_LICENSE_KEY`):
 *   NEW_RELIC_LICENSE_KEY — the 40-char ingest licence key (starts eu01xx for
 *   this account), NOT an NRAK-... user API key.  Without it reporting is off,
 *   exactly like SENTRY_DSN gates Sentry.
 */

// The account is in New Relic's EU region, so ingest must go to the EU
// collector — the US host silently rejects EU licence keys.
const EVENT_API_HOST = "https://insights-collector.eu01.nr-data.net";

const ACCOUNT_ID = "3352365";

// Tags every event, so this worker stays distinguishable in an account shared
// with timetable-api-node and timetable-pulse (cf. SENTRY_SERVICE_TAG).
const SERVICE_TAG = "gtfs-eta-worker";

/**
 * POST one custom event.  Never throws and never rejects — reporting must not
 * be able to break the worker.  In fetch, call it through ctx.waitUntil; in
 * scheduled, awaiting it is fine (nothing is waiting on the cron).
 */
export async function recordEvent(env, eventType, attributes) {
  const key = env.NEW_RELIC_LICENSE_KEY ?? "";
  if (!key) return;

  const accountId = env.NEW_RELIC_ACCOUNT_ID ?? ACCOUNT_ID;
  const url = `${EVENT_API_HOST}/v1/accounts/${accountId}/events`;

  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Api-Key": key,
        "Content-Type": "application/json",
      },
      body: JSON.stringify([{
        eventType,
        service: SERVICE_TAG,
        commit: env.GIT_COMMIT ?? null,
        ...attributes,
      }]),
      signal: AbortSignal.timeout(5000),
    });
    if (resp.status >= 300) {
      console.error(`[newrelic] event POST failed: HTTP ${resp.status}`);
    }
  } catch (exc) {
    console.error(`[newrelic] failed to record ${eventType}: ${exc}`);
  }
}
