"""Pack raw/ days older than the retrain window into archive/raw/<day>.tar.xz.

~8.6k small protobufs a day become one xz tar ~8x smaller; the originals are
deleted only after the uploaded archive round-trips. src.snapshots readers
fall back to the archive, so --all and old-day scoring still work.

    python scripts/pack_raw.py               # every raw/ day older than 45 days
    python scripts/pack_raw.py --day 2026-09-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

sys.path.insert(0, ".")

from src.snapshots import _make_client, archive_day, list_snapshot_days  # noqa: E402

# Retrain reads the last 36 days (retrain.yml LABEL_DAYS); keep slack past it.
DEFAULT_OLDER_THAN = 45


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--older-than", type=int, default=DEFAULT_OLDER_THAN,
                        help=f"pack days older than this many days (default {DEFAULT_OLDER_THAN})")
    parser.add_argument("--day", action="append", metavar="YYYY-MM-DD",
                        help="pack only this day (repeatable); still subject to --older-than")
    args = parser.parse_args()

    cutoff = (date.today() - timedelta(days=args.older_than)).isoformat()
    days = [d for d in (args.day or list_snapshot_days(archived=False)) if d < cutoff]
    print(f"{len(days)} day(s) before {cutoff} to pack")
    client = _make_client()
    for day in days:
        print(f"  {day}: {archive_day(client, day)} snapshots", flush=True)


if __name__ == "__main__":
    main()
