"""
Validate off-route detection on one day of real snapshots.

Produces:
  1. Summary: how many vehicles flagged off-route, when, for how long.
  2. A CSV of flagged events for manual inspection.
  3. Console output of a few detailed vehicle timelines.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from src.gtfs_static import get_gtfs
from src.snapshots import list_snapshot_keys, _make_client, _fetch_and_parse
from src.trip_inference import infer_trips

DATE = "2026-05-19"   # Monday — busy weekday, depot runs expected in evening
HOURS_UTC = range(14, 22)  # 17:00-01:00 Lviv time (peak + evening service)

print(f"Loading GTFS static…")
gtfs = get_gtfs()

print(f"Loading snapshots for {DATE} hours {list(HOURS_UTC)[0]}–{list(HOURS_UTC)[-1]} UTC…")
keys = list_snapshot_keys(date_str=DATE)
# Filter to selected hours (key format: raw/YYYY-MM-DD/YYYY-MM-DDTHH:MM:SSZ.pb)
hour_keys = [k for k in keys if int(k.split("T")[1][:2]) in HOURS_UTC]
print(f"  {len(hour_keys)} keys")

client = _make_client()
rows = []
with ThreadPoolExecutor(max_workers=12) as ex:
    for r in ex.map(lambda k: _fetch_and_parse(client, k), hour_keys):
        rows.extend(r)

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.dropna(subset=["lat", "lon"])
print(f"  {len(df):,} rows, {df['vehicle_id'].nunique()} vehicles")

print("Running trip inference + off-route detection…")
df = infer_trips(df, gtfs)

n_off = df["off_route"].sum()
n_total = len(df)
print(f"\nOff-route snapshots: {n_off:,} / {n_total:,} ({100*n_off/n_total:.1f}%)")
print(f"Unique vehicles ever off-route: {df[df['off_route']]['vehicle_id'].nunique()}")

# --- Per-vehicle off-route events ---
events = []
for vid, grp in df.groupby("vehicle_id"):
    grp = grp.sort_values("timestamp")
    in_event = False
    start = None
    for _, row in grp.iterrows():
        if row["off_route"] and not in_event:
            in_event = True
            start = row["timestamp"]
        elif not row["off_route"] and in_event:
            in_event = False
            events.append({
                "vehicle_id": vid,
                "route_id": row.get("route_id"),
                "off_route_start": start,
                "off_route_end": row["timestamp"],
                "duration_min": (row["timestamp"] - start).total_seconds() / 60,
            })
    if in_event:  # still off-route at end of window
        events.append({
            "vehicle_id": vid,
            "route_id": grp.iloc[-1].get("route_id"),
            "off_route_start": start,
            "off_route_end": None,
            "duration_min": None,
        })

events_df = pd.DataFrame(events)
if not events_df.empty:
    print(f"\nOff-route events detected: {len(events_df)}")
    print(f"Duration distribution (min):")
    print(events_df["duration_min"].describe().round(1))
    events_df.to_csv(f"data/off_route_events_{DATE}.csv", index=False)
    print(f"Saved → data/off_route_events_{DATE}.csv")

    # --- Detailed timeline for 3 vehicles with longest off-route periods ---
    top = events_df.sort_values("duration_min", ascending=False).head(3)
    print(f"\n--- Top 3 off-route vehicles (by duration) ---")
    for _, ev in top.iterrows():
        vid = ev["vehicle_id"]
        vdf = df[df["vehicle_id"] == vid].sort_values("timestamp")
        print(f"\nVehicle {vid}  route={ev['route_id']}  off-route from {ev['off_route_start']} for {ev['duration_min']:.0f} min")
        # Show last 3 on-route, all off-route, first 3 back on-route
        status_changes = vdf["off_route"].ne(vdf["off_route"].shift()).cumsum()
        for seg_id, seg in vdf.groupby(status_changes):
            label = "OFF" if seg["off_route"].iloc[0] else "ON "
            start_t = seg["timestamp"].iloc[0].strftime("%H:%M")
            end_t   = seg["timestamp"].iloc[-1].strftime("%H:%M")
            n = len(seg)
            print(f"  {label}  {start_t}–{end_t}  ({n} snapshots)")
else:
    print("\nNo off-route events detected in this window.")
    print("(This is expected if you picked a window with no depot runs.)")
