"""
Daily live-ETA quality job (run by .github/workflows/score-quality.yml).

Scores one completed UTC day — joining the archived `predictions/` feed against
the actual arrivals derived from `raw/` (see src/scoring.py) — and publishes the
report to R2:

    quality/YYYY-MM-DD.json   immutable per-day history
    quality/latest.json       newest report (cheap to poll)
    quality/latest.md         human-readable summary

Usage:
    python scripts/score_quality.py                 # score yesterday (UTC)
    python scripts/score_quality.py --date 2026-06-15

The worker's daily cron dispatches this workflow at 02:15 UTC, by which point
the previous UTC day is fully archived (Lviv's service day closes ~21:00 UTC).
This script owns scoring + publication, then runs the AI diagnosis
(scripts/diagnose.py) and the numeric auto-retrain gate (src/gate.py).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone

import boto3
import requests
from dotenv import load_dotenv

sys.path.insert(0, ".")

from src.gate import DEFAULT_WINDOW_DAYS, evaluate_gate  # noqa: E402
from src.scoring import score_date  # noqa: E402

load_dotenv()

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ.get("R2_BUCKET", "gtfs-lviv")

QUALITY_PREFIX = "quality/"
RETRAIN_MARKER_KEY = QUALITY_PREFIX + "_last_retrain.txt"
RETRAIN_WORKFLOW = "retrain.yml"
# Don't retrain more often than this — a retrain is a heavy ~hour-long job and
# the model only drifts slowly. Guards against a sustained breach re-triggering
# every day while the new model is still propagating.
GATE_COOLDOWN_DAYS = int(os.environ.get("GATE_COOLDOWN_DAYS", "14"))


def _make_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def _put(client, key: str, body: bytes, content_type: str) -> None:
    client.put_object(Bucket=R2_BUCKET, Key=key, Body=body, ContentType=content_type)


def _report_exists(client, date_str: str) -> bool:
    """True if quality/<date>.json is already published (this day was scored)."""
    try:
        client.head_object(Bucket=R2_BUCKET, Key=f"{QUALITY_PREFIX}{date_str}.json")
        return True
    except client.exceptions.ClientError:
        return False


# --------------------------------------------------------------------------
# Auto-retrain gate (src.gate decides; this dispatches + tracks cooldown)
# --------------------------------------------------------------------------

def _load_recent_reports(client, window_days: int) -> list[dict]:
    """Load the most recent per-day `quality/YYYY-MM-DD.json` reports from R2."""
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=QUALITY_PREFIX):
        for o in page.get("Contents", []):
            name = o["Key"][len(QUALITY_PREFIX):]
            # YYYY-MM-DD.json — exclude latest.json / latest.md / _last_retrain.txt
            if len(name) == 15 and name.endswith(".json") and name[4] == "-" == name[7]:
                keys.append(o["Key"])
    out: list[dict] = []
    for key in sorted(keys)[-window_days:]:
        try:
            out.append(json.loads(client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()))
        except Exception:
            pass
    return out


def _last_retrain_date(client) -> date | None:
    try:
        raw = client.get_object(Bucket=R2_BUCKET, Key=RETRAIN_MARKER_KEY)["Body"].read()
        return date.fromisoformat(raw.decode().strip())
    except Exception:
        return None


def _dispatch_retrain(token: str, repo: str, reason: str) -> bool:
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{RETRAIN_WORKFLOW}/dispatches"
    resp = requests.post(
        url,
        headers={
            "authorization": f"Bearer {token}",
            "accept": "application/vnd.github+json",
            "x-github-api-version": "2022-11-28",
            "user-agent": "gtfs-eta-quality-bot",
        },
        json={"ref": "main", "inputs": {"reason": reason[:280]}},
        timeout=30,
    )
    if resp.status_code not in (204, 200):
        print(f"  [gate] retrain dispatch failed: HTTP {resp.status_code} — {resp.text}", flush=True)
        return False
    return True


def maybe_retrain(report: dict, client) -> None:
    """Evaluate the rolling gate and auto-dispatch a retrain if it's breached."""
    reports = _load_recent_reports(client, DEFAULT_WINDOW_DAYS)
    # The just-scored report may not be back from R2's read-after-write yet.
    if report.get("date") not in {r.get("date") for r in reports}:
        reports.append(report)

    result = evaluate_gate(reports)
    print(f"  [gate] {result.reason}", flush=True)
    if not result.breach:
        return

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        print("  [gate] breach detected but GITHUB_TOKEN/REPOSITORY unset — not dispatching", flush=True)
        return

    today = date.fromisoformat(str(report["date"]))
    last = _last_retrain_date(client)
    if last is not None and (today - last).days < GATE_COOLDOWN_DAYS:
        print(f"  [gate] breach, but last retrain {last} is within {GATE_COOLDOWN_DAYS}d cooldown", flush=True)
        return

    reason = f"auto-retrain: {result.reason} (as of {report['date']})"
    if _dispatch_retrain(token, repo, reason):
        _put(client, RETRAIN_MARKER_KEY, today.isoformat().encode(), "text/plain")
        print(f"  [gate] 🔁 retrain dispatched — {result.reason}", flush=True)
        try:
            from diagnose import comment_on_quality_issue

            comment_on_quality_issue(f"🔁 **Auto-retrain dispatched** — {result.reason}.")
        except Exception:
            pass


def _render_markdown(report: dict) -> str:
    """A terse human summary — the JSON is the source of truth."""
    date = report.get("date", "?")
    if report.get("status") != "ok":
        return f"# ETA quality — {date}\n\nStatus: **{report.get('status')}** (nothing scored).\n"

    o = report["overall"]
    lines = [
        f"# ETA quality — {date}",
        "",
        f"- Predictions scored: **{report['n_predictions_scored']:,}**  "
        f"(coverage {report['coverage_frac']:.0%} of {report['n_actual_arrivals']:,} arrivals)",
        f"- MAE **{o['mae_sec']:.0f}s**  ·  median **{o['median_ae_sec']:.0f}s**  ·  "
        f"p90 **{o['p90_ae_sec']:.0f}s**  ·  bias **{o['bias_sec']:+.0f}s** "
        f"({'late' if o['bias_sec'] >= 0 else 'early'})",
    ]
    an = report.get("arriving_now") or {}
    if an.get("within_window_frac") is not None:
        lines.append(
            f"- Arriving-now calibration: **{an['within_window_frac']:.0%}** of "
            f"≤60 s predictions actually arrived within the minute ({an['n']:,} preds)"
        )

    lines += ["", "## MAE by lead time", "", "| horizon | n | MAE | bias |", "|---|---:|---:|---:|"]
    for bucket, m in (report.get("by_lead_bucket") or {}).items():
        lines.append(f"| {bucket} | {m['n']:,} | {m['mae_sec']:.0f}s | {m['bias_sec']:+.0f}s |")

    worst = report.get("worst_routes") or []
    if worst:
        lines += ["", "## Worst routes (by MAE)", "", "| route | n | MAE | bias |", "|---|---:|---:|---:|"]
        for m in worst:
            lines.append(
                f"| {m['route_id']} | {m['n']:,} | {m['mae_sec']:.0f}s | {m['bias_sec']:+.0f}s |"
            )
    return "\n".join(lines) + "\n"


def publish(report: dict, client=None) -> None:
    client = client or _make_client()
    date = report["date"]
    payload = json.dumps(report, indent=2, default=str).encode()

    _put(client, f"{QUALITY_PREFIX}{date}.json", payload, "application/json")
    _put(client, f"{QUALITY_PREFIX}latest.json", payload, "application/json")
    _put(
        client,
        f"{QUALITY_PREFIX}latest.md",
        _render_markdown(report).encode(),
        "text/markdown; charset=utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Score + publish daily ETA quality.")
    parser.add_argument(
        "--date",
        help="UTC day YYYY-MM-DD (default: yesterday)",
        default=(datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat(),
    )
    parser.add_argument(
        "--no-publish", action="store_true", help="print the report but don't write to R2"
    )
    parser.add_argument(
        "--no-diagnose", action="store_true", help="skip the AI diagnosis + GitHub issue"
    )
    parser.add_argument(
        "--no-retrain", action="store_true", help="skip the auto-retrain gate"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-score even if quality/<date>.json already exists (overrides the "
        "idempotency guard that keeps the schedule backstop from duplicating the "
        "worker-dispatched run)",
    )
    args = parser.parse_args()

    # Idempotency guard: this workflow fires twice a day — once from the worker's
    # 02:15 UTC workflow_dispatch, once from the GitHub `schedule` backstop (which
    # lands hours late). Both default to scoring *yesterday*. Without this guard
    # the second run re-scores the same day, posts a duplicate "Updated diagnosis"
    # comment, and re-evaluates the retrain gate. If the day is already published,
    # the primary run did everything — the backstop should no-op.
    if not args.no_publish and not args.force and _report_exists(_make_client(), args.date):
        print(
            f"  quality/{args.date}.json already exists — already scored; "
            f"skipping (use --force to re-score).",
            flush=True,
        )
        return 0

    print(f"Scoring live ETA quality for {args.date}…", flush=True)
    report = score_date(args.date)

    summary = report.get("overall")
    if summary:
        print(
            f"  scored {report['n_predictions_scored']:,} predictions · "
            f"MAE {summary['mae_sec']:.0f}s · bias {summary['bias_sec']:+.0f}s · "
            f"coverage {report['coverage_frac']:.0%}",
            flush=True,
        )
    else:
        print(f"  status: {report.get('status')}", flush=True)

    if not args.no_publish:
        publish(report)
        print(f"  published quality/{args.date}.json + latest.json + latest.md", flush=True)

    # AI diagnosis + GitHub issue (best-effort; both no-op without their creds).
    if not args.no_diagnose and report.get("status") == "ok":
        from diagnose import diagnose, upsert_issue

        diagnosis = diagnose(report)
        if diagnosis:
            url = upsert_issue(diagnosis, report)
            report["diagnosis"] = diagnosis
            report["diagnosis_issue"] = url
            if not args.no_publish:
                publish(report)  # republish so the JSON carries the diagnosis
            if diagnosis.get("recommend_retrain"):
                print("  [diagnose] AI recommends a retrain", flush=True)

    # Numeric auto-retrain gate (authoritative; advisory diagnosis is separate).
    if not args.no_retrain and report.get("status") == "ok":
        try:
            maybe_retrain(report, _make_client())
        except Exception as exc:
            print(f"  [gate] evaluation failed: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
