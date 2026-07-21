"""
Pull everything needed to review a day's live-ETA quality in one place.

The daily `score-quality.yml` job posts a metrics-only debug comment and
publishes the full report to R2. The *review* — analysing those metrics, making
the model/feature fixes, and leaving a round-summary comment for the next cycle —
is done on demand (interactively, or via `scripts/diagnose.py` for the AI read).

This helper gathers the two inputs that review needs and prints them:

    1. the full quality report  (quality/<date>.json on R2 — richer than the
       issue comment: full by_route, full by_hour, coverage_gap by_hour, …)
    2. the human engineering notes already on the rolling issue (prior
       round-summaries + architectural decisions) — the context for "what was
       already tried / must not be reverted"

Usage:
    python scripts/review_quality.py                 # review yesterday (UTC)
    python scripts/review_quality.py --date 2026-06-20
    make review-quality [DATE=2026-06-20]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

sys.path.insert(0, ".")

load_dotenv()


def _gh_token() -> str | None:
    """GITHUB_TOKEN/GH_TOKEN if set, else fall back to the local `gh` CLI."""
    import os

    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        return subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        ).stdout.strip() or None
    except Exception:
        return None


def _repo() -> str | None:
    """GITHUB_REPOSITORY if set, else ask the local `gh` CLI for owner/name."""
    import os

    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo
    try:
        return subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            capture_output=True, text=True, check=True,
        ).stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Gather a day's quality report + issue notes.")
    parser.add_argument(
        "--date",
        default=(datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat(),
        help="UTC day YYYY-MM-DD (default: yesterday)",
    )
    args = parser.parse_args()

    # 1. Full quality report from R2.
    from diagnose import _fetch_report  # reuses the same R2 client/bucket

    report = _fetch_report(args.date)
    if not report:
        return 1

    print(f"\n{'='*78}\n  QUALITY REPORT — {args.date}\n{'='*78}")
    print(json.dumps(report, indent=2, default=str))

    # 2. Human engineering notes on the rolling issue (prior rounds + decisions).
    from diagnose import _fetch_issue_comments

    token, repo = _gh_token(), _repo()
    print(f"\n{'='*78}\n  ENGINEERING NOTES (rolling quality issue)\n{'='*78}")
    if not token or not repo:
        print("  (no GitHub token/repo — set GITHUB_TOKEN or run `gh auth login`)")
        return 0

    notes = _fetch_issue_comments(token, repo)
    if not notes:
        print("  (no human notes yet)")
        return 0
    for c in notes:
        author = c.get("user", {}).get("login", "?")
        ts = c.get("created_at", "")[:10]
        print(f"\n--- [{ts} @{author}] {'-'*48}\n{c.get('body', '').strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
