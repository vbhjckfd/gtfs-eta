"""
End-to-end pipeline: R2 snapshots → trip inference → snapshot-anchored
training rows → parquet (data/training/).

Usage:
    python scripts/run_pipeline.py --date 2026-05-19
    python scripts/run_pipeline.py --all              # process all available days
    python scripts/run_pipeline.py --all --parallel 4 # 4 days at a time
    python scripts/run_pipeline.py --date 2026-05-19 --date 2026-05-20

Days are independent, and a single day is dominated by single-threaded CPU
work (trip inference + shape projection), so --parallel N processes N days in
separate worker processes. Each worker loads GTFS static once (from the local
cache pickle). On a 16 GB machine keep N ≤ 4-5: a worker peaks at ~2 GB.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.snapshots import list_snapshot_days, list_snapshot_keys, _make_client, _fetch_and_parse
from src.trip_inference import infer_trips
from src.labeling import build_training_rows
from src.live_features import side_paths, write_side_tables

from concurrent.futures import ThreadPoolExecutor
import pandas as pd

OUT_DIR = Path("data/training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Share of snapshots kept as training rows (whole snapshots, every horizon).
# A full day is ~3M rows held as Python dicts while it is built — the reason
# CI runs one day at a time — and the trainer samples far below that anyway.
DEFAULT_KEEP_PCT = 100.0


def _day_done(date_str: str, side: bool) -> bool:
    if not (OUT_DIR / f"{date_str}.parquet").exists():
        return False
    return not side or all(p.exists() for p in side_paths(OUT_DIR, date_str))


def process_day(date_str: str, gtfs, client, max_workers: int = 12,
                quiet: bool = False, keep_pct: float = DEFAULT_KEEP_PCT,
                side: bool = True) -> dict:
    # In parallel mode the inline progress fragments from several workers
    # would interleave mid-line, so quiet workers only report via the
    # per-day completion line printed by the parent.
    say = (lambda *a, **k: None) if quiet else print

    out_path = OUT_DIR / f"{date_str}.parquet"
    if _day_done(date_str, side):
        say(f"  {date_str}: already done, skipping")
        return {"date": date_str, "status": "skipped"}

    t0 = time.time()
    say(f"  {date_str}: listing keys…", end=" ", flush=True)
    keys = list_snapshot_keys(date_str=date_str)
    if not keys:
        say("no keys, skipping")
        return {"date": date_str, "status": "no_keys"}
    say(f"{len(keys)} keys", end=" | ", flush=True)

    # --- Load snapshots ---
    say("downloading…", end=" ", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(lambda k: _fetch_and_parse(client, k), keys):
            rows.extend(r)

    df = pd.DataFrame(rows)
    del rows
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["lat", "lon"])
    say(f"{len(df):,} rows ({df['vehicle_id'].nunique()} vehicles)", end=" | ", flush=True)
    feed_cols = [c for c in ("vehicle_ts", "odometer") if c in df.columns]
    feed = df[["vehicle_id", "timestamp", "speed"] + feed_cols].copy() if side else None
    df = df.drop(columns=feed_cols)

    # --- Trip inference ---
    say("inferring trips…", end=" ", flush=True)
    df = infer_trips(df, gtfs)
    n_high = df["high_confidence"].sum()
    n_low  = (~df["high_confidence"]).sum()
    say(f"high_conf={n_high:,} low_conf={n_low:,}", end=" | ", flush=True)

    # --- Build snapshot-anchored training rows ---
    say("building training rows…", end=" ", flush=True)
    positions: list | None = [] if side else None
    training = build_training_rows(df, gtfs, trip_col="inferred_trip_id",
                                   positions_out=positions, keep_pct=keep_pct)
    if training.empty:
        say("no training rows generated!")
        return {"date": date_str, "status": "empty_training"}

    if side:
        write_side_tables(OUT_DIR, date_str, feed, positions)
    training.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    say(f"{len(training):,} training rows → {out_path.name}  [{elapsed:.0f}s]")
    return {
        "date": date_str,
        "status": "ok",
        "n_snapshots": len(keys),
        "n_rows": len(df),
        "n_labeled": len(training),
        "elapsed_s": elapsed,
    }


# --- Parallel workers -------------------------------------------------------
# Initialised once per worker process; the boto3 client cannot cross process
# boundaries. GTFS static is loaded per day, not once — see
# get_gtfs_for_date: a static change mid-range must label each day against
# the schedule actually in effect that day, not whatever the worker process
# happened to load first.

_worker_client = None


def _init_worker():
    global _worker_client
    _worker_client = _make_client()


def _process_day_in_worker(date_str: str, keep_pct: float, side: bool) -> dict:
    from src.gtfs_static import get_gtfs_for_date
    gtfs = get_gtfs_for_date(date_str, client=_worker_client)
    return process_day(date_str, gtfs, _worker_client, quiet=True,
                       keep_pct=keep_pct, side=side)


def _run_parallel(days: list[str], n_workers: int, keep_pct: float, side: bool) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    print(f"Processing {len(days)} days with {n_workers} workers…")
    results = []
    # One day per worker process: a day's peak heap is never handed back to the
    # OS, so a reused worker carries the previous day's high-water mark into
    # the next one.
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             max_tasks_per_child=1) as ex:
        futures = {ex.submit(_process_day_in_worker, d, keep_pct, side): d for d in days}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if r["status"] == "ok":
                print(f"  {r['date']}: {r['n_labeled']:,} training rows  [{r['elapsed_s']:.0f}s]")
            else:
                print(f"  {r['date']}: {r['status']}")
    results.sort(key=lambda r: r["date"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", action="append", dest="dates", metavar="YYYY-MM-DD")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="process N days concurrently (default: 1)")
    parser.add_argument("--since", metavar="YYYY-MM-DD",
                        help="with --all: only days on or after this date")
    parser.add_argument("--until", metavar="YYYY-MM-DD",
                        help="with --all: only days on or before this date")
    parser.add_argument("--keep-pct", type=float, default=DEFAULT_KEEP_PCT,
                        help="share of snapshots kept as training rows (default: all)")
    parser.add_argument("--no-side", dest="side", action="store_false",
                        help="skip the side tables the live features need")
    args = parser.parse_args()

    if args.all:
        # Discover all available days from R2 (one LIST per day prefix, not
        # one per snapshot key).
        days = list_snapshot_days()
        if args.since:
            days = [d for d in days if d >= args.since]
        if args.until:
            days = [d for d in days if d <= args.until]
        print(f"Found {len(days)} days: {days[0]} → {days[-1]}")
    elif args.dates:
        days = sorted(args.dates)
    else:
        parser.print_help()
        sys.exit(1)

    todo = [d for d in days if not _day_done(d, args.side)]
    if len(todo) < len(days):
        print(f"  {len(days) - len(todo)} day(s) already done")
    days = todo
    if not days:
        print("Nothing to do.")
        return

    if args.parallel > 1 and len(days) > 1:
        results = _run_parallel(days, min(args.parallel, len(days)), args.keep_pct, args.side)
    else:
        from src.gtfs_static import get_gtfs_for_date
        client = _make_client()
        results = [
            process_day(d, get_gtfs_for_date(d, client=client), client,
                        keep_pct=args.keep_pct, side=args.side)
            for d in days
        ]

    print("\n=== Summary ===")
    for r in results:
        if r["status"] == "ok":
            print(f"  {r['date']}: {r['n_labeled']:,} training rows  ({r['elapsed_s']:.0f}s)")
        else:
            print(f"  {r['date']}: {r['status']}")


if __name__ == "__main__":
    main()
