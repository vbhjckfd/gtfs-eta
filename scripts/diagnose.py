"""
AI diagnosis of the daily ETA-quality report (step 4 of the feedback loop).

Feeds the scored metrics + worst-case exemplars to Claude (claude-opus-4-8) and
asks for a structured, ranked diagnosis — each finding tied to a concrete model
lever (a feature, a hyperparameter, the off-route filter, schedule drift, …) —
then upserts a single GitHub issue so findings accumulate in one place instead
of one issue per day.

Both stages degrade gracefully: no ANTHROPIC_API_KEY → skip the LLM call; no
GitHub token/repo → skip the issue. Scoring + publication (score_quality.py)
never depends on either succeeding.
"""

from __future__ import annotations

import json
import os
from typing import Literal

import requests

MODEL = "claude-opus-4-8"
ISSUE_TITLE = "📉 Live ETA quality — automated diagnosis"
ISSUE_LABEL = "eta-quality"

# The model's job is to map error patterns to fixes; keep the response bounded.
_MAX_TOKENS = 8000
_HTTP_TIMEOUT = 30


# --------------------------------------------------------------------------
# LLM diagnosis
# --------------------------------------------------------------------------

def _fetch_issue_comments(token: str, repo: str) -> list[dict]:
    """Return all human engineering notes from the rolling quality issue (best-effort).

    Excludes automated github-actions one-liners ("Updated diagnosis for…") since
    they just restate the metrics already in the prompt and crowd out the real
    architectural context.  Fetches up to 100 comments (paginated if needed).
    """
    api = f"https://api.github.com/repos/{repo}/issues"
    try:
        issues = _gh("GET", f"{api}?state=open&per_page=100", token)
        issues.raise_for_status()
        match = next((i for i in issues.json()
                      if i.get("title") == ISSUE_TITLE and "pull_request" not in i), None)
        if not match:
            return []
        all_comments: list[dict] = []
        page = 1
        while True:
            resp = _gh("GET", f"{api}/{match['number']}/comments?per_page=100&page={page}", token)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_comments.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        # Drop automated bot comments — they restate metrics already in the prompt.
        return [
            c for c in all_comments
            if c.get("user", {}).get("login") != "github-actions"
        ]
    except Exception:
        return []


def _build_prompt(report: dict, issue_comments: list[dict] | None = None) -> str:
    """A compact, fully-contextualised prompt — metrics + system background."""
    metrics = {
        k: report.get(k)
        for k in (
            "date", "n_predictions_scored", "n_actual_arrivals", "coverage_frac",
            "overall", "by_lead_bucket", "by_hour", "by_stops_ahead",
            "arriving_now", "worst_routes", "coverage_gap",
        )
    }

    comments_block = ""
    if issue_comments:
        lines = ["\n\nEngineering notes from this issue (ALL of them — read carefully "
                 "before diagnosing; they document architectural decisions, confirmed "
                 "root causes, and changes that are already in production):"]
        for c in issue_comments:
            author = c.get("user", {}).get("login", "?")
            ts     = c.get("created_at", "")[:10]
            body   = c.get("body", "").strip()[:4000]
            lines.append(f"\n[{ts} @{author}]\n{body}")
        comments_block = "\n".join(lines)

    return (
        "You are diagnosing the live accuracy of a GTFS-RT arrival-time prediction "
        "feed for Lviv, Ukraine public transport. A gradient-boosted model "
        "(sklearn HistGradientBoostingRegressor) predicts seconds-to-arrival per "
        "upcoming stop, anchored at each vehicle position snapshot. Features: "
        "route_id, stop_sequence, stops_ahead, hour, day_of_week, month, "
        "is_weekend, is_holiday, remaining_dist_m, progress_speed_mps, "
        "stops_remaining, trip_progress_frac, dist_per_stop_m, "
        "speed_eta_warm (remaining_dist / effective_speed — warm-started with "
        "route+hour historical median when current speed is unknown), "
        "hist_speed_mps (route+hour historical median speed), "
        "hist_travel_time_est ((stops_ahead - 1) × historical p65 seconds-per-stop — "
        "accumulates intermediate stop dwell; the imminent stop is excluded to avoid "
        "double-counting with speed_eta_warm). "
        "CRITICAL architectural decisions — DO NOT suggest reverting these: "
        "(1) sched_remaining_sec (GTFS schedule) removed — in-person verification "
        "confirmed stop times are unreliable in Lviv, especially at terminus. "
        "(2) hist_travel_time_est uses (stops_ahead-1)*tps, NOT stops_ahead*tps — "
        "the stops_ahead=1 formula was intentionally changed to 0 to fix +44 s "
        "pessimism at the imminent stop. "
        "(3) hist_tps uses p65 (not median) to reduce long-horizon optimism. "
        "The baseline is speed_eta_warm.\n\n"
        "Below is one day of scored quality, joining what the feed *predicted* "
        "against what *actually* happened (derived from raw vehicle positions). "
        "error_sec is signed: positive = predicted arrival LATER than reality "
        "(pessimistic); negative = predicted EARLIER (optimistic). lead buckets are "
        "the horizon the rider saw. coverage_frac is the share of real arrivals "
        "that received any prediction.\n\n"
        f"{json.dumps(metrics, indent=2, default=str)}"
        f"{comments_block}\n\n"
        "Diagnose the most actionable problems. For each finding: tie it to a "
        "specific lever (a named feature, a hyperparameter, the off-route/"
        "trip-matching filter, an infra/coverage gap, or training data), give a "
        "concrete hypothesis for the cause, and a specific suggested change. Rank "
        "by impact. Do not invent metrics not present above; if the data is "
        "insufficient for a confident call, say so and lower confidence. "
        "Recommend a retrain only for systemic degradation a retrain would plausibly "
        "fix (drift, broad bias), not for a single route or hour."
    )


def diagnose(report: dict) -> dict | None:
    """Return a structured diagnosis dict, or None if unavailable/failed."""
    if report.get("status") != "ok":
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [diagnose] ANTHROPIC_API_KEY not set — skipping AI diagnosis", flush=True)
        return None

    try:
        import anthropic
        from pydantic import BaseModel, Field
    except ImportError:
        print("  [diagnose] anthropic/pydantic not installed (pip install -e '.[ai]')", flush=True)
        return None

    class Finding(BaseModel):
        title: str = Field(description="One-line summary of the problem")
        severity: Literal["low", "medium", "high"]
        lever: Literal[
            "feature", "hyperparameter", "training-data",
            "off-route", "infra", "schedule-drift", "other",
        ]
        hypothesis: str = Field(description="Likely cause, grounded in the metrics")
        suggested_action: str = Field(description="A specific, concrete change")
        confidence: Literal["low", "medium", "high"]

    class Diagnosis(BaseModel):
        summary: str = Field(description="2-3 sentence overall read of the day")
        findings: list[Finding]
        recommend_retrain: bool
        retrain_reason: str

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo  = os.environ.get("GITHUB_REPOSITORY")
    issue_comments: list[dict] = []
    if token and repo:
        issue_comments = _fetch_issue_comments(token, repo)
        if issue_comments:
            print(f"  [diagnose] loaded {len(issue_comments)} engineering notes for context", flush=True)

    client = anthropic.Anthropic()
    try:
        resp = client.messages.parse(
            model=MODEL,
            max_tokens=_MAX_TOKENS,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": _build_prompt(report, issue_comments)}],
            output_format=Diagnosis,
        )
    except Exception as exc:  # never let a model/API hiccup fail the job
        print(f"  [diagnose] Claude call failed: {exc}", flush=True)
        return None

    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        print(f"  [diagnose] no structured output (stop_reason={resp.stop_reason})", flush=True)
        return None

    return resp.parsed_output.model_dump()


# --------------------------------------------------------------------------
# GitHub issue upsert
# --------------------------------------------------------------------------

def _signed(v) -> str:
    """`+12`/`-7`, or `?` when the metric is missing (never raises on None)."""
    return f"{v:+g}" if isinstance(v, (int, float)) else "?"


def _late_early(bias) -> str:
    return "late" if isinstance(bias, (int, float)) and bias >= 0 else "early"


def _findings_md(diagnosis: dict) -> list[str]:
    """The ranked findings, most-severe first — the heart of the diagnosis."""
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    findings = sorted(diagnosis.get("findings", []), key=lambda f: sev_rank.get(f["severity"], 3))
    if not findings:
        return ["_No actionable findings this run._"]
    lines: list[str] = []
    for f in findings:
        lines += [
            f"### [{f['severity'].upper()}] {f['title']}  ·  `{f['lever']}`  (confidence: {f['confidence']})",
            f"- **Cause:** {f['hypothesis']}",
            f"- **Fix:** {f['suggested_action']}",
            "",
        ]
    return lines


def _render_comment(report: dict, diagnosis: dict | None = None) -> str:
    """The deeply technical per-day record posted as an issue comment.

    Metrics-only by default (what `score-quality.yml` posts daily — no AI call):
    the full error breakdown the body omits — lead-time, hour, stops-ahead,
    arriving-now calibration, worst routes, coverage gaps. When a `diagnosis` is
    supplied (the on-demand `scripts/diagnose.py` run), the AI summary, retrain
    flag, and ranked findings are appended. Each comment is permanent, so every
    scored day accumulates in one timeline.
    """
    o = report.get("overall") or {}
    L: list[str] = [f"## 🔬 ETA quality — {report.get('date')}", ""]

    if diagnosis:
        summary = (diagnosis.get("summary") or "").strip()
        if summary:
            L += [summary, ""]

    L.append(
        f"**Headline** · MAE **{o.get('mae_sec')}s** · median **{o.get('median_ae_sec')}s** · "
        f"p90 **{o.get('p90_ae_sec')}s** · bias **{_signed(o.get('bias_sec'))}s** "
        f"({_late_early(o.get('bias_sec'))}) · coverage **{report.get('coverage_frac', 0):.0%}** "
        f"({report.get('n_predictions_scored', 0):,} preds / {report.get('n_actual_arrivals', 0):,} arrivals)"
    )
    if diagnosis and diagnosis.get("recommend_retrain"):
        L.append(f"- ⚠️ **Retrain recommended:** {(diagnosis.get('retrain_reason') or '').strip()}")
    rj = report.get("relaxed_join") or {}
    rjo = rj.get("overall") or {}
    if rjo:
        # Rider-centric upper bound: match on physical (vehicle, stop_id) + nearest
        # time, ignoring trip-relative trip_id/stop_sequence. The gap between this
        # and the strict coverage is the join-artifact share; the rest is a real
        # gap (issue #3: only ~3.4pp is artifact).
        L.append(
            f"- _Rider-centric join (physical stop, nearest time) · MAE "
            f"**{rjo.get('mae_sec')}s** · coverage **{rj.get('coverage_frac', 0):.0%}** · "
            f"{rj.get('n_predictions_scored', 0):,} preds — upper bound; the strict↔this "
            f"coverage gap is join artifact, the remainder a real gap._"
        )
    L.append("")

    # Error by lead time — the horizon the rider actually saw (most informative).
    by_lead = report.get("by_lead_bucket") or {}
    if by_lead:
        L += ["### Error by lead time (rider's horizon)", "",
              "| horizon | n | MAE | median | p90 | bias |", "|---|---:|---:|---:|---:|---:|"]
        for bucket, m in by_lead.items():
            L.append(f"| {bucket} | {m['n']:,} | {m['mae_sec']:.0f}s | {m['median_ae_sec']:.0f}s "
                     f"| {m['p90_ae_sec']:.0f}s | {_signed(m['bias_sec'])}s |")
        L.append("")

    # Error by stops-ahead — separates near-stop dwell error from long-horizon drift.
    by_sa = report.get("by_stops_ahead") or {}
    if by_sa:
        ordered = sorted(by_sa.items(), key=lambda kv: int(kv[0]))[:15]
        L += ["### Error by stops ahead", "",
              "| stops ahead | n | MAE | bias |", "|---:|---:|---:|---:|"]
        for sa, m in ordered:
            L.append(f"| {sa} | {m['n']:,} | {m['mae_sec']:.0f}s | {_signed(m['bias_sec'])}s |")
        L.append("")

    # Worst hours by MAE — surfaces peak-load or off-hours degradation.
    by_hour = {h: m for h, m in (report.get("by_hour") or {}).items() if m["n"] >= 20}
    if by_hour:
        worst_hours = sorted(by_hour.items(), key=lambda kv: kv[1]["mae_sec"], reverse=True)[:6]
        L += ["### Worst hours by MAE (n ≥ 20)", "",
              "| hour (feed TZ) | n | MAE | bias |", "|---:|---:|---:|---:|"]
        for h, m in worst_hours:
            L.append(f"| {int(h):02d}:00 | {m['n']:,} | {m['mae_sec']:.0f}s | {_signed(m['bias_sec'])}s |")
        L.append("")

    an = report.get("arriving_now") or {}
    if an.get("within_window_frac") is not None:
        L += [f"### Arriving-now calibration\n\n**{an['within_window_frac']:.0%}** of ≤60 s "
              f"predictions actually arrived within the minute ({an['n']:,} preds).", ""]

    worst = report.get("worst_routes") or []
    if worst:
        L += ["### Worst routes by MAE", "",
              "| route | n | MAE | median | p90 | bias |", "|---|---:|---:|---:|---:|---:|"]
        for m in worst:
            L.append(f"| {m['route_id']} | {m['n']:,} | {m['mae_sec']:.0f}s | {m['median_ae_sec']:.0f}s "
                     f"| {m['p90_ae_sec']:.0f}s | {_signed(m['bias_sec'])}s |")
        L.append("")

    # Coverage gaps — which arrivals got no prediction at all (feed/pipeline holes).
    cov_gap = report.get("coverage_gap") or {}
    cov_routes = cov_gap.get("by_route") or {}
    by_cause = cov_gap.get("by_cause")
    gappy = [(r, g) for r, g in cov_routes.items() if g["n_uncovered"] > 0][:8]
    if gappy:
        # Cause columns appear once the scorer is fed the served feed (the cause
        # tagging distinguishes a vehicle never served / matched to the wrong trip
        # / stop beyond the served horizon).
        has_cause = any("uncovered_by_cause" in g for _, g in gappy)
        if by_cause:
            tot = sum(by_cause.values()) or 1
            order = ["vehicle_absent", "trip_mismatch", "stop_missing"]
            parts = [f"{c.replace('_', ' ')} {by_cause.get(c, 0):,} ({by_cause.get(c, 0) / tot:.0%})"
                     for c in order if by_cause.get(c)]
            L += [f"_Uncovered arrivals by cause: {' · '.join(parts)}._", ""]
        if has_cause:
            L += ["### Coverage gaps (most uncovered arrivals)", "",
                  "| route | actual | uncovered | coverage | absent | mismatch | missing |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
            for r, g in gappy:
                c = g.get("uncovered_by_cause") or {}
                L.append(f"| {r} | {g['n_actual']:,} | {g['n_uncovered']:,} | {g['coverage_frac']:.0%} "
                         f"| {c.get('vehicle_absent', 0):,} | {c.get('trip_mismatch', 0):,} "
                         f"| {c.get('stop_missing', 0):,} |")
        else:
            L += ["### Coverage gaps (most uncovered arrivals)", "",
                  "| route | actual | uncovered | coverage |", "|---|---:|---:|---:|"]
            for r, g in gappy:
                L.append(f"| {r} | {g['n_actual']:,} | {g['n_uncovered']:,} | {g['coverage_frac']:.0%} |")
        L.append("")

    if diagnosis:
        L += ["## Findings", ""]
        L += _findings_md(diagnosis)
    L += ["---",
          "<sub>Per-day metrics from `score-quality.yml`"
          + ("; AI findings from `scripts/diagnose.py`" if diagnosis else "")
          + ". Full JSON: `quality/" + str(report.get("date")) + ".json` on R2.</sub>"]
    return "\n".join(L)


def _render_issue_body(report: dict, diagnosis: dict | None = None) -> str:
    o = report.get("overall") or {}
    lines = [
        f"_Automated daily quality — latest run scored **{report.get('date')}**._",
        "",
    ]
    if diagnosis and diagnosis.get("summary"):
        lines += [f"**{diagnosis['summary'].strip()}**", ""]
    lines.append(
        f"- MAE **{o.get('mae_sec')}s** · median **{o.get('median_ae_sec')}s** · "
        f"bias **{_signed(o.get('bias_sec'))}s** · coverage **{report.get('coverage_frac', 0):.0%}** "
        f"({report.get('n_predictions_scored', 0):,} predictions)"
    )
    if diagnosis and diagnosis.get("recommend_retrain"):
        lines.append(f"- ⚠️ **Retrain recommended:** {diagnosis.get('retrain_reason', '').strip()}")
    if diagnosis:
        lines += ["", "## Findings", ""]
        lines += _findings_md(diagnosis)
    lines += [
        "---",
        "<sub>Generated by `score-quality.yml`. Body reflects the most recent run; "
        "the full per-day metrics breakdown is in each day's comment below, and the "
        "raw JSON in `quality/DATE.json` on R2.</sub>",
    ]
    return "\n".join(lines)


def _gh(method: str, url: str, token: str, **kwargs) -> requests.Response:
    headers = {
        "authorization": f"Bearer {token}",
        "accept": "application/vnd.github+json",
        "x-github-api-version": "2022-11-28",
        "user-agent": "gtfs-eta-quality-bot",
    }
    return requests.request(method, url, headers=headers, timeout=_HTTP_TIMEOUT, **kwargs)


def upsert_issue(report: dict, diagnosis: dict | None = None) -> str | None:
    """Create or update the single rolling quality issue. Returns its URL or None.

    The issue body is the at-a-glance dashboard (latest run); the deeply technical
    per-day record is posted as a comment so each scored day accumulates in one
    timeline. Metrics-only when `diagnosis` is omitted (the daily CI run); AI
    findings are appended when it's supplied (the on-demand diagnose run).
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")  # "owner/name" (set in Actions)
    if not token or not repo:
        print("  [diagnose] GITHUB_TOKEN/GITHUB_REPOSITORY not set — skipping issue", flush=True)
        return None

    body = _render_issue_body(report, diagnosis)
    comment = _render_comment(report, diagnosis)
    api = f"https://api.github.com/repos/{repo}/issues"

    try:
        existing = _gh("GET", f"{api}?state=open&per_page=100", token)
        existing.raise_for_status()
        match = next((i for i in existing.json()
                      if i.get("title") == ISSUE_TITLE and "pull_request" not in i), None)

        if match:
            num = match["number"]
            _gh("PATCH", f"{api}/{num}", token, json={"body": body}).raise_for_status()
            _gh("POST", f"{api}/{num}/comments", token, json={"body": comment}).raise_for_status()
            print(f"  [diagnose] updated issue #{num}", flush=True)
            return match["html_url"]

        created = _gh("POST", api, token, json={
            "title": ISSUE_TITLE, "body": body, "labels": [ISSUE_LABEL],
        })
        created.raise_for_status()
        num = created.json()["number"]
        _gh("POST", f"{api}/{num}/comments", token, json={"body": comment}).raise_for_status()
        url = created.json()["html_url"]
        print(f"  [diagnose] opened {url}", flush=True)
        return url
    except Exception as exc:
        print(f"  [diagnose] GitHub issue upsert failed: {exc}", flush=True)
        return None


def _fetch_report(date_str: str) -> dict | None:
    """Load a previously-published quality report (quality/<date>.json) from R2."""
    import sys

    sys.path.insert(0, ".")
    from src.snapshots import R2_BUCKET, _make_client  # noqa: E402

    client = _make_client()
    try:
        obj = client.get_object(Bucket=R2_BUCKET, Key=f"quality/{date_str}.json")
        return json.loads(obj["Body"].read())
    except Exception as exc:
        print(f"  [diagnose] could not load quality/{date_str}.json: {exc}", flush=True)
        return None


def main() -> int:
    """On-demand AI diagnosis: load a published day, run Claude, post the findings.

    The daily `score-quality.yml` job posts metrics-only comments (no AI). Run
    this manually when you want a structured AI read of a specific day:

        python scripts/diagnose.py                 # diagnose yesterday (UTC)
        python scripts/diagnose.py --date 2026-06-20
    """
    import argparse
    from datetime import datetime, timedelta, timezone

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the AI diagnosis on a scored day.")
    parser.add_argument(
        "--date",
        default=(datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat(),
        help="UTC day YYYY-MM-DD (default: yesterday)",
    )
    parser.add_argument(
        "--no-issue", action="store_true", help="print the diagnosis but don't touch the issue"
    )
    args = parser.parse_args()

    report = _fetch_report(args.date)
    if not report:
        return 1
    if report.get("status") != "ok":
        print(f"  [diagnose] {args.date} status={report.get('status')} — nothing to diagnose", flush=True)
        return 0

    diagnosis = diagnose(report)
    if not diagnosis:
        return 1

    print(json.dumps(diagnosis, indent=2), flush=True)
    if not args.no_issue:
        url = upsert_issue(report, diagnosis)
        if url:
            print(f"  [diagnose] posted AI diagnosis to {url}", flush=True)
    return 0


def comment_on_quality_issue(text: str) -> bool:
    """Post a comment on the open rolling quality issue, if one exists."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        return False
    api = f"https://api.github.com/repos/{repo}/issues"
    try:
        existing = _gh("GET", f"{api}?state=open&per_page=100", token)
        existing.raise_for_status()
        match = next((i for i in existing.json()
                      if i.get("title") == ISSUE_TITLE and "pull_request" not in i), None)
        if not match:
            return False
        _gh("POST", f"{api}/{match['number']}/comments", token, json={"body": text}).raise_for_status()
        return True
    except Exception as exc:
        print(f"  [diagnose] issue comment failed: {exc}", flush=True)
        return False


if __name__ == "__main__":
    import sys

    sys.exit(main())
