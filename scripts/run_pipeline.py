"""
End-to-end pipeline: R2 snapshots → trip inference → labeling → parquet.

Usage:
    python scripts/run_pipeline.py --date 2026-05-19
    python scripts/run_pipeline.py --all          # process all available days
    python scripts/run_pipeline.py --date 2026-05-19 --date 2026-05-20
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.gtfs_static import get_gtfs
from src.snapshots import list_snapshot_keys, _make_client, _fetch_and_parse
from src.trip_inference import infer_trips
from src.labeling import build_labels

from concurrent.futures import ThreadPoolExecutor
import pandas as pd

OUT_DIR = Path("data/labeled")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def process_day(date_str: str, gtfs, client, max_workers: int = 12) -> dict:
    out_path = OUT_DIR / f"{date_str}.parquet"
    if out_path.exists():
        print(f"  {date_str}: already done, skipping")
        return {"date": date_str, "status": "skipped"}

    t0 = time.time()
    print(f"  {date_str}: listing keys…", end=" ", flush=True)
    keys = list_snapshot_keys(date_str=date_str)
    if not keys:
        print("no keys, skipping")
        return {"date": date_str, "status": "no_keys"}
    print(f"{len(keys)} keys", end=" | ", flush=True)

    # --- Load snapshots ---
    print("downloading…", end=" ", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(lambda k: _fetch_and_parse(client, k), keys):
            rows.extend(r)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["lat", "lon"])
    print(f"{len(df):,} rows ({df['vehicle_id'].nunique()} vehicles)", end=" | ", flush=True)

    # --- Trip inference ---
    print("inferring trips…", end=" ", flush=True)
    df = infer_trips(df, gtfs)
    n_high = df["high_confidence"].sum()
    n_low  = (~df["high_confidence"]).sum()
    print(f"high_conf={n_high:,} low_conf={n_low:,}", end=" | ", flush=True)

    # --- Build labels ---
    print("labeling…", end=" ", flush=True)
    labeled = build_labels(df, gtfs, trip_col="inferred_trip_id")
    if labeled.empty:
        print("no labels generated!")
        return {"date": date_str, "status": "empty_labels"}

    labeled.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    print(f"{len(labeled):,} labeled rows → {out_path.name}  [{elapsed:.0f}s]")
    return {
        "date": date_str,
        "status": "ok",
        "n_snapshots": len(keys),
        "n_rows": len(df),
        "n_labeled": len(labeled),
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", action="append", dest="dates", metavar="YYYY-MM-DD")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    print("Loading GTFS static…")
    gtfs = get_gtfs()
    client = _make_client()

    if args.all:
        # Discover all available days from R2
        from collections import defaultdict
        from src.snapshots import list_snapshot_keys as list_keys
        all_keys = list_keys()
        days = sorted({k.split("/")[1] for k in all_keys if k.count("/") >= 2})
        print(f"Found {len(days)} days: {days[0]} → {days[-1]}")
    elif args.dates:
        days = sorted(args.dates)
    else:
        parser.print_help()
        sys.exit(1)

    results = []
    for d in days:
        results.append(process_day(d, gtfs, client))

    print("\n=== Summary ===")
    for r in results:
        if r["status"] == "ok":
            print(f"  {r['date']}: {r['n_labeled']:,} labeled rows  ({r['elapsed_s']:.0f}s)")
        else:
            print(f"  {r['date']}: {r['status']}")


if __name__ == "__main__":
    main()
