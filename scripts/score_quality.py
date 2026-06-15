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
The AI diagnosis and auto-retrain gate are layered on in later steps; this
script owns scoring + publication.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import boto3
from dotenv import load_dotenv

sys.path.insert(0, ".")

from src.scoring import score_date  # noqa: E402

load_dotenv()

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ.get("R2_BUCKET", "gtfs-lviv")

QUALITY_PREFIX = "quality/"


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
    args = parser.parse_args()

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
