"""
Per-route MAE digest (run by .github/workflows/route-mae.yml at 22:00 UTC).

22:00 UTC = 01:00 Kyiv, by which point the Kyiv day that just ended is closed
(Lviv's service day stops ~21:00 UTC) but score-quality.yml — which fires at
02:15 UTC — hasn't run yet. So this job does the scoring itself, publishes the
report to R2 exactly like score_quality.py does, and then opens a *fresh* GitHub
issue holding the full per-route MAE table plus its day-over-day delta, so
there's something concrete to read over breakfast.

Because it publishes quality/<date>.json, the 02:15 score-quality run finds the
day already scored and no-ops via its idempotency guard — this job effectively
takes over primary scoring, four hours earlier. It therefore also posts the
rolling-issue metrics comment that score_quality.py used to own.

Usage:
    python scripts/route_mae.py                  # score the UTC day that just closed
    python scripts/route_mae.py --date 2026-07-20
    python scripts/route_mae.py --force          # re-score + re-post an already-scored day
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(__file__))

from diagnose import _fetch_report, _gh, ISSUE_LABEL, upsert_issue  # noqa: E402
from score_quality import _make_client, _report_exists, publish  # noqa: E402
from src.scoring import score_date  # noqa: E402

load_dotenv()

# Routes with fewer scored predictions than this are grouped into a tail line —
# their MAE swings on a handful of trips and reads as noise in the table.
MIN_N = 30


def _issue_title(date_str: str) -> str:
    return f"📊 Route MAE — {date_str}"


def _delta(cur: float, prev: float | None) -> str:
    """Signed day-over-day change, or an em dash when yesterday had no sample."""
    if prev is None:
        return "—"
    return f"{cur - prev:+.0f}s"


def render_issue(report: dict, prev: dict | None) -> str:
    """The full per-route table, worst MAE first, with a Δ vs the previous day."""
    date_str = report["date"]
    o = report["overall"]
    prev_by_route = (prev or {}).get("by_route") or {}
    prev_overall = (prev or {}).get("overall") or {}

    lines = [
        f"Scored **{report['n_predictions_scored']:,}** predictions "
        f"({report['coverage_frac']:.0%} coverage of "
        f"{report['n_actual_arrivals']:,} actual arrivals).",
        "",
        f"Overall MAE **{o['mae_sec']:.0f}s** "
        f"(Δ {_delta(o['mae_sec'], prev_overall.get('mae_sec'))} vs previous day) · "
        f"median **{o['median_ae_sec']:.0f}s** · p90 **{o['p90_ae_sec']:.0f}s** · "
        f"bias **{o['bias_sec']:+.0f}s** "
        f"({'late' if o['bias_sec'] >= 0 else 'early'}).",
        "",
        "## MAE by route",
        "",
        "Worst first. Δ compares against the same route the previous day; "
        f"routes with fewer than {MIN_N} scored predictions are summarised at the end.",
        "",
        "| route | n | MAE | Δ MAE | median | p90 | bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    by_route = report.get("by_route") or {}
    ranked = sorted(
        ((r, m) for r, m in by_route.items() if m["n"] >= MIN_N),
        key=lambda kv: kv[1]["mae_sec"],
        reverse=True,
    )
    for route, m in ranked:
        prev_mae = (prev_by_route.get(route) or {}).get("mae_sec")
        lines.append(
            f"| {route} | {m['n']:,} | {m['mae_sec']:.0f}s | "
            f"{_delta(m['mae_sec'], prev_mae)} | {m['median_ae_sec']:.0f}s | "
            f"{m['p90_ae_sec']:.0f}s | {m['bias_sec']:+.0f}s |"
        )

    thin = [m for m in by_route.values() if m["n"] < MIN_N]
    if thin:
        n_thin = sum(m["n"] for m in thin)
        lines.append(
            f"\n<sub>{len(thin)} route(s) with &lt;{MIN_N} predictions "
            f"({n_thin:,} rows total) omitted as too thin to read.</sub>"
        )

    if not ranked and not thin:
        lines.append("| _no routes scored_ | | | | | | |")

    lines += _horizon_section(report)
    lines += _worst_routes_horizon_section(report)
    lines += _worst_predictions_section(report)

    lines += [
        "",
        "---",
        "",
        f"<sub>Full report: `quality/{date_str}.json` in R2. "
        "Deeper AI read of this day: `make diagnose DATE=" + date_str + "`.</sub>",
    ]
    return "\n".join(lines) + "\n"


def _horizon_section(report: dict) -> list[str]:
    """Overall bias/MAE by stops_ahead. #5 was a bias that grew with horizon
    while the overall number stayed unremarkable — this is the early warning."""
    bsa = report.get("by_stops_ahead") or {}
    if not bsa:
        return []
    lines = ["", "## Bias by horizon (stops_ahead)", "",
             "Negative bias = optimistic (bus arrives later than predicted) — "
             "the failure mode from issue #5. Should stay near zero across the "
             "whole curve, not just in aggregate.", "",
             "| stops_ahead | n | bias | MAE |", "|---:|---:|---:|---:|"]
    for h in sorted(bsa, key=lambda k: int(k)):
        m = bsa[h]
        lines.append(f"| {h} | {m['n']:,} | {m['bias_sec']:+.0f}s | {m['mae_sec']:.0f}s |")
    return lines


def _worst_routes_horizon_section(report: dict) -> list[str]:
    """Per-horizon bias for the worst-MAE routes — catches a localized version
    of #5's regression on a single route before it shows up in the aggregate."""
    worst = [r for r in (report.get("worst_routes") or []) if r.get("by_stops_ahead")]
    if not worst:
        return []
    lines = ["", "## Worst routes — bias by horizon", ""]
    for r in worst[:5]:
        bsa = r["by_stops_ahead"]
        cells = [
            f"sa={h}: {bsa[h]['bias_sec']:+.0f}s" for h in sorted(bsa, key=lambda k: int(k))
        ]
        lines.append(f"- **route {r['route_id']}** ({r['mae_sec']:.0f}s MAE): " + " · ".join(cells))
    return lines


def _worst_predictions_section(report: dict) -> list[str]:
    """The single worst individual predictions of the day — outlier exemplars
    a per-route MAE average can hide."""
    worst = report.get("worst_predictions") or []
    if not worst:
        return []
    lines = ["", "## Worst individual predictions", "",
             "| route | stop | stops_ahead | lead | error |",
             "|---|---|---:|---:|---:|"]
    for p in worst:
        lines.append(
            f"| {p['route_id']} | {p['stop_id']} | {p['stops_ahead']} | "
            f"{p['lead_sec'] / 60:.0f}m | {p['error_sec']:+.0f}s |"
        )
    return lines


def post_issue(report: dict, prev: dict | None) -> str | None:
    """Open (or update, if re-run) the day's route-MAE issue. Best-effort."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        print("  [route-mae] GITHUB_TOKEN/GITHUB_REPOSITORY not set — skipping issue", flush=True)
        return None

    title = _issue_title(report["date"])
    body = render_issue(report, prev)
    api = f"https://api.github.com/repos/{repo}/issues"

    try:
        # A --force re-run of the same day should rewrite its issue, not add a twin.
        existing = _gh("GET", f"{api}?state=all&per_page=100", token)
        existing.raise_for_status()
        match = next((i for i in existing.json()
                      if i.get("title") == title and "pull_request" not in i), None)
        if match:
            _gh("PATCH", f"{api}/{match['number']}", token,
                json={"body": body}).raise_for_status()
            print(f"  [route-mae] updated issue #{match['number']}", flush=True)
            return match["html_url"]

        created = _gh("POST", api, token,
                      json={"title": title, "body": body, "labels": [ISSUE_LABEL]})
        created.raise_for_status()
        url = created.json()["html_url"]
        print(f"  [route-mae] opened {url}", flush=True)
        return url
    except Exception as exc:
        print(f"  [route-mae] GitHub issue post failed: {exc}", flush=True)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a day and post its per-route MAE issue.")
    parser.add_argument(
        "--date",
        # At 22:00 UTC the *current* UTC date is the Kyiv day that just ended —
        # not "yesterday" as in the 02:15 UTC jobs.
        default=datetime.now(timezone.utc).date().isoformat(),
        help="UTC day YYYY-MM-DD (default: the day that just closed)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-score and re-post even if quality/<date>.json already exists",
    )
    parser.add_argument(
        "--no-publish", action="store_true", help="don't write the report to R2"
    )
    parser.add_argument(
        "--no-issue", action="store_true", help="print the digest instead of posting it"
    )
    args = parser.parse_args()

    # Same guard score_quality.py uses: if the day is already published, some other
    # run owned it — don't re-score and don't post a second issue.
    if not args.force and not args.no_publish and _report_exists(_make_client(), args.date):
        print(f"  quality/{args.date}.json already exists — skipping (use --force).", flush=True)
        return 0

    print(f"Scoring live ETA quality for {args.date}…", flush=True)
    report = score_date(args.date)

    if report.get("status") != "ok":
        print(f"  status: {report.get('status')} — nothing to report.", flush=True)
        if not args.no_publish:
            publish(report)
        return 0

    o = report["overall"]
    print(
        f"  scored {report['n_predictions_scored']:,} predictions · "
        f"MAE {o['mae_sec']:.0f}s · bias {o['bias_sec']:+.0f}s · "
        f"coverage {report['coverage_frac']:.0%}",
        flush=True,
    )

    if not args.no_publish:
        publish(report)
        print(f"  published quality/{args.date}.json + latest.json + latest.md", flush=True)

    prev = _fetch_report((date.fromisoformat(args.date) - timedelta(days=1)).isoformat())

    if args.no_issue:
        print(render_issue(report, prev))
        return 0

    url = post_issue(report, prev)
    # Keep the rolling diagnosis issue fed: this job now scores the day before
    # score-quality.yml gets to it, so it inherits that job's metrics comment too.
    upsert_issue(report)
    if url and not args.no_publish:
        report["route_mae_issue"] = url
        publish(report)  # republish so the JSON carries the issue link
    return 0


if __name__ == "__main__":
    sys.exit(main())
