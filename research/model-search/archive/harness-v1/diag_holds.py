"""Mid-trip holds (stationary >= 10 min, away from the trip ends) per day for given routes (run 20).

  python research/model-search/diag_holds.py 2026-09-07..2026-09-21 133 122

Asks whether the long holds that dominate route 133 / 122 errors recur at the same place and clock
time day after day (a driver-break timetable a feature could learn) or are random.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/model-search")
from harness import expand_days  # noqa: E402

days, routes = expand_days(sys.argv[1]), sys.argv[2:]
rows = []
for d in days:
    tr = pd.read_parquet(f"data/ms_lite/{d}.parquet", columns=["trip_id", "route_id"]).drop_duplicates()
    tr = tr[tr.route_id.astype(str).isin(routes)]
    p = pd.read_parquet(f"data/ms_lite/{d}.pos.parquet").merge(tr, on="trip_id")
    p = p.sort_values(["vehicle_id", "snapshot_ts"])
    for (veh, trip), g in p.groupby(["vehicle_id", "trip_id"]):
        x = g.dist_along_m.to_numpy()
        t = g.snapshot_ts.astype("int64").to_numpy() / 10**(9 if g.snapshot_ts.dt.unit == "ns" else 6)
        L = np.nanmax(x)
        moved = np.r_[True, np.abs(np.diff(x)) > 15]
        grp = np.cumsum(moved)
        for k in np.unique(grp):
            m = grp == k
            dur = t[m][-1] - t[m][0]
            if dur >= 600 and 300 < x[m][0] < L - 300:
                lt = pd.Timestamp(t[m][0], unit="s", tz="UTC").tz_convert("Europe/Kyiv")
                rows.append(dict(day=d, dow=lt.day_name()[:3], route=g.route_id.iloc[0], veh=veh, trip=trip,
                                 start=lt.strftime("%H:%M"), min=round(dur / 60), at_m=round(x[m][0] / 100) * 100))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for r in routes:
    s = df[df.route.astype(str) == r]
    print(f"\n=== route {r}: {len(s)} mid-trip holds >= 10 min over {len(days)} days")
    print(s.drop(columns="route").to_string(index=False))
