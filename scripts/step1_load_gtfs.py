"""Step 1 — Download GTFS static feed and verify the parse."""
import sys
sys.path.insert(0, ".")

from src.gtfs_static import get_gtfs

print("Loading GTFS static feed (download + parse + cache)…")
gtfs = get_gtfs(force_download=True, force_rebuild=True)

routes = gtfs.all_routes()
trips  = gtfs.all_trips()

print(f"\n  Routes:  {len(routes)}")
print(f"  Trips:   {len(trips)}")
print(f"  Stops:   {len(gtfs._stops)}")
print(f"  Shapes:  {len(gtfs._shapes)}")

# Spot-check a random trip
import random
sample_tid = random.choice(list(gtfs._trip_index.keys()))
info = gtfs.get_trip(sample_tid)
shape = gtfs.get_shape_linestring(info.shape_id)
print(f"\nSample trip: {sample_tid}")
print(f"  route_id={info.route_id}  shape_id={info.shape_id}  stops={len(info.stop_times)}")
if shape:
    print(f"  shape length: {shape.length:.0f} m  (projected EPSG:32635)")
else:
    print("  ⚠ no shape")

# Confirm stop distance caching worked
with_dist = sum(1 for (sid, _) in gtfs._stop_distances)
print(f"\nStop-distance cache entries: {len(gtfs._stop_distances)}")
print("Step 1 complete ✓")
