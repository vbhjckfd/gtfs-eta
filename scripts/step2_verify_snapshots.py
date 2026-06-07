"""
Step 2 — Verify R2 snapshot reading.
Load one full day of snapshots and report basic stats.
"""
import sys
sys.path.insert(0, ".")

from datetime import datetime, timezone
from src.snapshots import load_snapshots_df, list_snapshot_keys

# Use a full mid-range day
DATE = "2026-05-17"
print(f"Loading snapshots for {DATE}…")

keys = list_snapshot_keys(date_str=DATE)
print(f"  Keys found: {len(keys)}")

# Load just the first 60 keys (~1 hour) to keep this fast
sample_keys = keys[:60]
print(f"  Parsing first {len(sample_keys)} snapshots…")

from src.snapshots import _make_client, _fetch_and_parse
from concurrent.futures import ThreadPoolExecutor
client = _make_client()
all_rows = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(_fetch_and_parse, client, k) for k in sample_keys]
    for f in futures:
        all_rows.extend(f.result())

import pandas as pd
df = pd.DataFrame(all_rows)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

print(f"\n  Total rows:       {len(df)}")
print(f"  Unique vehicles:  {df['vehicle_id'].nunique()}")
print(f"  Unique routes:    {df['route_id'].nunique()}")
print(f"  Unique trips:     {df['trip_id'].nunique()}")
print(f"  Time range:       {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"  Bearing non-null: {df['bearing'].notna().sum()} / {len(df)}")
print(f"  Speed non-null:   {df['speed'].notna().sum()} / {len(df)}")
print(f"  route_id null:    {df['route_id'].isna().sum()} / {len(df)}")
print(f"  trip_id null:     {df['trip_id'].isna().sum()} / {len(df)}")
print("\nSample rows:")
print(df[["timestamp","vehicle_id","route_id","trip_id","lat","lon","bearing","speed"]].head(5).to_string(index=False))
print("\nStep 2 complete ✓")
