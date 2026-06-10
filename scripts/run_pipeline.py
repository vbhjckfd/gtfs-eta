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

from src.snapshots import list_snapshot_keys, _make_client, _fetch_and_parse
from src.trip_inference import infer_trips
from src.labeling import build_training_rows

from concurrent.futures import ThreadPoolExecutor
import pandas as pd

OUT_DIR = Path("data/training")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def process_day(date_str: str, gtfs, client, max_workers: int = 12,
                quiet: bool = False) -> dict:
    # In parallel mode the inline progress fragments from several workers
    # would interleave mid-line, so quiet workers only report via the
    # per-day completion line printed by the parent.
    say = (lambda *a, **k: None) if quiet else print

    out_path = OUT_DIR / f"{date_str}.parquet"
    if out_path.exists():
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["lat", "lon"])
    say(f"{len(df):,} rows ({df['vehicle_id'].nunique()} vehicles)", end=" | ", flush=True)

    # --- Trip inference ---
    say("inferring trips…", end=" ", flush=True)
    df = infer_trips(df, gtfs)
    n_high = df["high_confidence"].sum()
    n_low  = (~df["high_confidence"]).sum()
    say(f"high_conf={n_high:,} low_conf={n_low:,}", end=" | ", flush=True)

    # --- Build snapshot-anchored training rows ---
    say("building training rows…", end=" ", flush=True)
    training = build_training_rows(df, gtfs, trip_col="inferred_trip_id")
    if training.empty:
        say("no training rows generated!")
        return {"date": date_str, "status": "empty_training"}

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
# Initialised once per worker process; GTFS static loads from the cache
# pickle, the boto3 client cannot cross process boundaries.

_worker_gtfs = None
_worker_client = None


def _init_worker():
    global _worker_gtfs, _worker_client
    from src.gtfs_static import get_gtfs
    _worker_gtfs = get_gtfs()
    _worker_client = _make_client()


def _process_day_in_worker(date_str: str) -> dict:
    return process_day(date_str, _worker_gtfs, _worker_client, quiet=True)


def _run_parallel(days: list[str], n_workers: int) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    print(f"Processing {len(days)} days with {n_workers} workers…")
    results = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        futures = {ex.submit(_process_day_in_worker, d): d for d in days}
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
    args = parser.parse_args()

    if args.all:
        # Discover all available days from R2
        all_keys = list_snapshot_keys()
        days = sorted({k.split("/")[1] for k in all_keys if k.count("/") >= 2})
        print(f"Found {len(days)} days: {days[0]} → {days[-1]}")
    elif args.dates:
        days = sorted(args.dates)
    else:
        parser.print_help()
        sys.exit(1)

    if args.parallel > 1 and len(days) > 1:
        results = _run_parallel(days, min(args.parallel, len(days)))
    else:
        from src.gtfs_static import get_gtfs
        print("Loading GTFS static…")
        gtfs = get_gtfs()
        client = _make_client()
        results = [process_day(d, gtfs, client) for d in days]

    print("\n=== Summary ===")
    for r in results:
        if r["status"] == "ok":
            print(f"  {r['date']}: {r['n_labeled']:,} training rows  ({r['elapsed_s']:.0f}s)")
        else:
            print(f"  {r['date']}: {r['status']}")


if __name__ == "__main__":
    main()
