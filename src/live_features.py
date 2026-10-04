"""
Live network / per-vehicle state features for the ETA model (feature set
``live_v2``).

The production model used to see one snapshot of one vehicle: where it is,
how fast it moved since the last push, how long it has been stopped. The
research search on branch ``claude/model-search`` (runs 1-29) found most of
the remaining error is information that exists *across* snapshots and
vehicles, and measured ≈ -24% held-out MAE for the recipe built here:

  * live link store — for every stop-to-stop link, the last few traversal
    times by any vehicle (routes sharing a street share its traffic), summed
    along the path to the target stop;
  * historical link × local-hour medians from the previous 14 days (static,
    built at export like the route×hour priors);
  * the vehicle's own previous traversals of the path links (its previous
    lap) and how slow it has been running against history in the last hour;
  * conditional remaining dwell at its current location given how long it
    has already been stopped (static table of prior days' stops);
  * the feed's own per-vehicle fields: GPS-fix age, reported speed, odometer.

One engine, two callers. :class:`LiveState` is fed the same two streams in
training and in serving — every vehicle's raw feed fields
(:meth:`LiveState.observe_feed`) and every on-route vehicle's projected
position on its trip (:meth:`LiveState.observe_position`) — and answers
:meth:`LiveState.features` for a vehicle and its upcoming stops. The trainer
replays a day's side tables (written by scripts/run_pipeline.py) through it;
the push-feed daemon calls it from src/inference.run_inference and persists
it across its 5-minute restarts. Parity between the two is by construction,
not by keeping two implementations in sync.

Causality: anything detected from other vehicles is only visible from the
*next* snapshot on (``t_avail < t``), because the daemon processes a push's
vehicles in feed order and must not depend on it.
"""

from __future__ import annotations

import math
from bisect import bisect_right, insort
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SIDE_DIR = "side"

FEED_TZ = ZoneInfo("Europe/Kyiv")

# ── Windows (seconds) ─────────────────────────────────────────────────────
LINK_WINDOW_SEC = 1800       # newest traversal of a link must be this recent
LINK_KEEP = 8                # traversals kept per link (median of last 3 / 5)
LINK_MAX_SEC = 1800          # a single link traversal longer than this is a gap
LAP_WINDOW_SEC = 3 * 3600    # own previous traversal of a link
VEH_WINDOW_SEC = 3600        # own vs historical link times
POS_KEEP_SEC = 360           # own position ring (speed over <= 300 s)
FEED_KEEP_SEC = 2400         # feed ring (odometer over <= 1200 s, tolerance 1200 s)
RUN_RESET_FRAC = 0.5         # labeling._RUN_RESET_FRAC
RUN_RESET_CONSEC = 3         # labeling._RUN_RESET_CONSEC
FIRST_FIX_STOP_TOL_M = 50.0  # labeling._FIRST_FIX_STOP_TOL_M

HIST_MIN_N = 3               # historical link medians need this many traversals
DWELL_BIN_M = 50.0
DWELL_MIN_N = 5
DWELL_MOVE_EPS_M = 25.0      # labeling._MOVE_EPS_M
DWELL_MAX_GAP_SEC = 120.0
HIST_DAYS = 14

# Order is the model's: src/train_live.py and src/inference.py index by it.
LIVE_COLS = [
    "live_path_sec_m3", "live_path_cov", "live_path_age", "live_speed_mps",
    "own_speed_60", "own_speed_180", "own_speed_300", "live_path_sec_m5",
    "hist_path_sec", "hist_path_cov", "hist_path_sec_dt",
    "cur_link_live", "cur_link_hist", "cur_link_frac",
    "lap_path_sec", "lap_path_cov", "lap_path_age", "lap_fill_path_sec",
    "veh_hist_ratio60", "veh_hist_n60",
    "dwell_rem_med", "dwell_rem_p75", "dwell_p_more120", "dwell_n", "dwell_long_share",
    "pos_age_s", "pos_age_med600", "pos_age_dist",
    "fs_now", "fs_mean60", "odo_spd_60", "odo_spd_180", "odo_m_300",
    "odo_spd_600", "odo_spd_1200", "fs_zero_frac600",
]
N_LIVE = len(LIVE_COLS)
_COL = {c: i for i, c in enumerate(LIVE_COLS)}
# The exported trees carry no NaN handling for numeric splits: every missing
# live value is this sentinel (measured better than NaN, research run 1).
MISSING = -1.0

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Trip geometry: a trip's stops sorted along its shape
# ---------------------------------------------------------------------------

class TripGeom:
    """Stops of one trip in shape order, plus the link keys between them."""

    __slots__ = ("dists", "sids", "keys", "length")

    def __init__(self, dists: list[float], sids: list[str]):
        self.dists = dists
        self.sids = sids
        # keys[j] is the link arriving at stop j (keys[0] unused).
        self.keys = [""] + [f"{sids[j - 1]}>{sids[j]}" for j in range(1, len(sids))]
        self.length = dists[-1] if dists else 0.0


def _geom_from_entries(entries) -> TripGeom | None:
    if not entries:
        return None
    return TripGeom([float(e[0]) for e in entries], [str(e[1]) for e in entries])


def geom_from_worker_data(data: dict):
    """Trip geometry callable over the exported GTFS dict (serving)."""
    from src.inference import _trip_stop_entries

    cache: dict = {}

    def geom(trip_id: str) -> TripGeom | None:
        g = cache.get(trip_id, False)
        if g is False:
            g = _geom_from_entries(_trip_stop_entries(trip_id, data))
            cache[trip_id] = g
        return g

    return geom


def geom_from_gtfs(gtfs):
    """The same geometry over a GTFSStatic (training) — mirrors
    src/inference._trip_stop_entries on the exported dict exactly."""
    cache: dict = {}
    stop_dist = gtfs._stop_distances

    def geom(trip_id: str) -> TripGeom | None:
        g = cache.get(trip_id, False)
        if g is False:
            trip = gtfs.get_trip(str(trip_id))
            if trip is None:
                g = None
            else:
                entries = sorted(
                    (stop_dist.get((trip.shape_id, st.stop_id), 0.0), st.stop_id, st.stop_sequence, i)
                    for i, st in enumerate(trip.stop_times)
                )
                g = _geom_from_entries(entries)
            cache[trip_id] = g
        return g

    return geom


def loc_key(g: TripGeom, d: float) -> str:
    """Dwell location: last passed stop and the 50 m bin of the offset past it."""
    n = len(g.dists)
    a = min(max(bisect_right(g.dists, d), 1), n) - 1
    off = max(d - g.dists[a], 0.0) // DWELL_BIN_M
    return f"{g.sids[a]}@{int(off)}"


# ---------------------------------------------------------------------------
# Static tables (built from prior days at training and at export)
# ---------------------------------------------------------------------------

def local_hour_weekend(t: float) -> tuple[int, int]:
    lt = datetime.fromtimestamp(t, tz=timezone.utc).astimezone(FEED_TZ)
    return lt.hour, int(lt.weekday() >= 5)


def _local_hw_arrays(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = pd.to_datetime(pd.Series(t, dtype=float), unit="s", utc=True).dt.tz_convert(FEED_TZ)
    return s.dt.hour.to_numpy(), (s.dt.dayofweek.to_numpy() >= 5).astype(np.int8)


class LiveTables:
    """Historical link medians and the dwell table.

    hist_h  : link key -> float[24]      median traversal per local hour (NaN = < HIST_MIN_N)
    hist_k  : link key -> float          median over all hours
    hist_dt : link key -> float[2, 24]   per (weekend flag, local hour)
    dwell   : location key -> sorted float[]  durations of prior stationary runs there
    """

    def __init__(self, hist_h=None, hist_k=None, hist_dt=None, dwell=None):
        self.hist_h = hist_h or {}
        self.hist_k = hist_k or {}
        self.hist_dt = hist_dt or {}
        self.dwell = dwell or {}

    def hist(self, key: str, h: int) -> float:
        arr = self.hist_h.get(key)
        if arr is not None:
            v = arr[h]
            if v == v:
                return float(v)
        return self.hist_k.get(key, _NAN)

    def hist_d(self, key: str, wk: int, h: int, fallback: float) -> float:
        arr = self.hist_dt.get(key)
        if arr is not None:
            v = arr[wk, h]
            if v == v:
                return float(v)
        return fallback

    def to_dict(self) -> dict:
        return {"hist_h": self.hist_h, "hist_k": self.hist_k,
                "hist_dt": self.hist_dt, "dwell": self.dwell}

    @classmethod
    def from_dict(cls, d: dict | None) -> "LiveTables":
        d = d or {}
        return cls(d.get("hist_h"), d.get("hist_k"), d.get("hist_dt"), d.get("dwell"))

    @classmethod
    def build(cls, links: list[pd.DataFrame], runs: list[pd.DataFrame]) -> "LiveTables":
        """From prior days' link traversals (key, t, lt) and stationary runs (key, t, D)."""
        hist_h: dict = {}
        hist_k: dict = {}
        hist_dt: dict = {}
        L = pd.concat([x for x in links if len(x)], ignore_index=True) if links else pd.DataFrame()
        if len(L):
            h, wk = _local_hw_arrays(L["t"].to_numpy(dtype=float))
            L = pd.DataFrame({"key": L["key"].astype(str).to_numpy(), "h": h, "wk": wk,
                              "lt": L["lt"].to_numpy(dtype=float)})
            g = L.groupby(["key", "h"])["lt"].agg(["median", "size"]).reset_index()
            g = g[g["size"] >= HIST_MIN_N]
            for key, sub in g.groupby("key", sort=False):
                arr = np.full(24, np.nan)
                arr[sub["h"].to_numpy()] = sub["median"].to_numpy()
                hist_h[key] = arr
            k = L.groupby("key")["lt"].agg(["median", "size"])
            k = k[k["size"] >= HIST_MIN_N]
            hist_k = {str(i): float(v) for i, v in k["median"].items()}
            g = L.groupby(["key", "wk", "h"])["lt"].agg(["median", "size"]).reset_index()
            g = g[g["size"] >= HIST_MIN_N]
            for key, sub in g.groupby("key", sort=False):
                arr = np.full((2, 24), np.nan)
                arr[sub["wk"].to_numpy(), sub["h"].to_numpy()] = sub["median"].to_numpy()
                hist_dt[key] = arr
        dwell: dict = {}
        R = pd.concat([x for x in runs if len(x)], ignore_index=True) if runs else pd.DataFrame()
        if len(R):
            R = R.sort_values(["key", "D"])
            for key, sub in R.groupby("key", sort=False):
                dwell[str(key)] = sub["D"].to_numpy(dtype=np.float32)
        return cls(hist_h, hist_k, hist_dt, dwell)


def stationary_runs(pos: pd.DataFrame, geom) -> pd.DataFrame:
    """A day's stationary runs keyed by location (the dwell table's raw rows).

    Same anchor rule as labeling._stationary_seconds (a vehicle has not moved
    until it clears 25 m), per vehicle and trip, with a gap over 120 s
    starting a new anchor: D = seconds from the anchor to the last snapshot
    before the vehicle moves on. Zero-length runs are dropped.
    """
    if pos.empty:
        return pd.DataFrame({"key": pd.Series(dtype=object), "t": [], "D": []})
    p = pd.DataFrame({"vt": pos["vehicle_id"].astype(str).to_numpy() + "|"
                      + pos["trip_id"].astype(str).to_numpy(),
                      "tid": pos["trip_id"].astype(str).to_numpy(),
                      "t": pos["t"].to_numpy(dtype=float),
                      "d": pos["dist_along_m"].to_numpy(dtype=float)}
                     ).drop_duplicates(["vt", "t"]).sort_values(["vt", "t"]).reset_index(drop=True)
    vt = p["vt"].to_numpy()
    t = p["t"].to_numpy()
    d = p["d"].to_numpy()
    anc = np.empty(len(p), dtype=np.int64)
    a = 0
    for i in range(len(p)):
        if i == 0 or vt[i] != vt[i - 1] or d[i] - d[a] > DWELL_MOVE_EPS_M or t[i] - t[i - 1] > DWELL_MAX_GAP_SEC:
            a = i
        anc[i] = a
    last = np.r_[anc[1:] != anc[:-1], True]
    ai = anc[last]
    D = t[last] - t[ai]
    ok = D > 0
    ai, D = ai[ok], D[ok]
    tids = p["tid"].to_numpy()
    keys = []
    for i in ai:
        g = geom(tids[i])
        keys.append(loc_key(g, d[i]) if g is not None else "")
    r = pd.DataFrame({"key": keys, "t": t[ai], "D": D})
    return r[r["key"] != ""].reset_index(drop=True)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class _Veh:
    __slots__ = ("feed", "pos", "trip", "geom", "furthest", "next_idx", "last_t", "last_d",
                 "cross_idx", "cross_t", "streak", "own", "vlinks", "last_feed_t", "last_pos_t")

    def __init__(self):
        self.feed: deque = deque()        # (t, vehicle_ts, speed, odometer_m)
        self.pos: deque = deque()         # (t, trip_id, d)
        self.trip: str | None = None
        self.geom: TripGeom | None = None
        self.furthest = 0.0
        self.next_idx = 0
        self.last_t = 0.0
        self.last_d = 0.0
        self.cross_idx = -1               # stop index of the last crossing on this run
        self.cross_t = 0.0
        self.streak = 0
        self.own: dict = {}               # link key -> (t_cross, t_avail, lt), any trip
        self.vlinks: deque = deque()      # (t_cross, t_avail, own lt, hist lt)
        self.last_feed_t = -1.0
        self.last_pos_t = -1.0


class LiveState:
    """Network + per-vehicle state, fed snapshot by snapshot in time order."""

    def __init__(self, tables: LiveTables | None = None, links_out: list | None = None):
        self.tables = tables or LiveTables()
        self.links: dict[str, list] = {}   # link key -> [(t_cross, t_avail, lt)] sorted by t_cross
        self.veh: dict[str, _Veh] = {}
        self.links_out = links_out          # collects (key, t_cross, lt, vehicle) when given
        self._hw_cache: tuple[float, int, int] = (-1.0, 0, 0)
        # trip_id -> TripGeom; set by the daemon (geom_from_worker_data)
        self.geom = None

    # ── inputs ────────────────────────────────────────────────────────────

    def _v(self, vid: str) -> _Veh:
        v = self.veh.get(vid)
        if v is None:
            v = self.veh[vid] = _Veh()
        return v

    def _hw(self, t: float) -> tuple[int, int]:
        c = self._hw_cache
        if c[0] != t:
            h, wk = local_hour_weekend(t)
            self._hw_cache = c = (t, h, wk)
        return c[1], c[2]

    def observe_feed(self, vid: str, t: float, vehicle_ts, speed, odometer_km) -> None:
        """The vehicle's raw feed fields at feed time *t* (every entity, before any filter)."""
        v = self._v(vid)
        if t <= v.last_feed_t:
            return                      # same upstream snapshot served twice
        v.last_feed_t = t
        vts = float(vehicle_ts) if vehicle_ts is not None and vehicle_ts == vehicle_ts else _NAN
        spd = float(speed) if speed is not None and speed == speed else _NAN
        odo = float(odometer_km) * 1000.0 if odometer_km is not None and odometer_km == odometer_km else _NAN
        q = v.feed
        q.append((t, vts, spd, odo))
        lim = t - FEED_KEEP_SEC
        while q and q[0][0] < lim:
            q.popleft()

    def observe_position(self, vid: str, trip_id: str, t: float, d: float, geom: TripGeom | None) -> None:
        """An on-route vehicle's projected distance along its matched trip at feed time *t*."""
        v = self._v(vid)
        if t <= v.last_pos_t:
            return
        v.last_pos_t = t
        q = v.pos
        q.append((t, trip_id, d))
        lim = t - POS_KEEP_SEC
        while q and q[0][0] < lim:
            q.popleft()
        if geom is None:
            v.trip = None
            return
        if v.trip != trip_id:
            self._start_run(v, trip_id, geom, t, d)
            return
        if d < v.furthest - RUN_RESET_FRAC * max(geom.length, 1.0):
            v.streak += 1
            if v.streak >= RUN_RESET_CONSEC:
                self._start_run(v, trip_id, geom, t, d)
            return
        v.streak = 0
        dists = geom.dists
        n = len(dists)
        while v.next_idx < n and dists[v.next_idx] <= d:
            j = v.next_idx
            sd = dists[j]
            if d > v.last_d and t > v.last_t and sd > v.last_d:
                tc = v.last_t + (sd - v.last_d) / (d - v.last_d) * (t - v.last_t)
            else:
                tc = t
            if v.cross_idx == j - 1 and j >= 1:
                lt = tc - v.cross_t
                if 0.0 < lt < LINK_MAX_SEC:
                    self._add_link(vid, v, geom.keys[j], tc, t, lt)
            v.cross_idx = j
            v.cross_t = tc
            v.next_idx = j + 1
        if d > v.furthest:
            v.furthest = d
        v.last_t = t
        v.last_d = d

    def _start_run(self, v: _Veh, trip_id: str, geom: TripGeom, t: float, d: float) -> None:
        v.trip = trip_id
        v.geom = geom
        v.furthest = d
        v.next_idx = bisect_right(geom.dists, d)
        v.last_t = t
        v.last_d = d
        v.cross_idx = -1
        v.streak = 0
        # labeling._FIRST_FIX_STOP_TOL_M: a run's first fix counts as crossing
        # the stop it stands at (within 50 m past it), so the first link out of
        # a terminus is timed from there.
        a = v.next_idx - 1
        if a >= 0 and d - geom.dists[a] <= FIRST_FIX_STOP_TOL_M:
            v.cross_idx = a
            v.cross_t = t

    def _add_link(self, vid: str, v: _Veh, key: str, tc: float, t_avail: float, lt: float) -> None:
        lst = self.links.get(key)
        if lst is None:
            lst = self.links[key] = []
        insort(lst, (tc, t_avail, lt))
        if len(lst) > LINK_KEEP:
            del lst[0]
        v.own[key] = (tc, t_avail, lt)
        h, _wk = self._hw(tc)
        hl = self.tables.hist(key, h)
        v.vlinks.append((tc, t_avail, lt, hl))
        lim = t_avail - VEH_WINDOW_SEC - 120
        while v.vlinks and v.vlinks[0][0] < lim:
            v.vlinks.popleft()
        if self.links_out is not None:
            self.links_out.append((key, tc, lt, vid))

    # ── features ──────────────────────────────────────────────────────────

    def _link_live(self, key: str, t: float):
        """(latest lt, median of last 3, median of last 5, age) of a link visible at t."""
        lst = self.links.get(key)
        if not lst:
            return None
        vis = [x for x in lst if x[1] < t and x[0] <= t]
        if not vis:
            return None
        tc, _ta, lt = vis[-1]
        if t - tc > LINK_WINDOW_SEC:
            return None
        l3 = sorted(x[2] for x in vis[-3:])
        l5 = sorted(x[2] for x in vis[-5:])
        m3 = l3[len(l3) // 2] if len(l3) % 2 else 0.5 * (l3[len(l3) // 2 - 1] + l3[len(l3) // 2])
        m5 = l5[len(l5) // 2] if len(l5) % 2 else 0.5 * (l5[len(l5) // 2 - 1] + l5[len(l5) // 2])
        return lt, m3, m5, t - tc

    def features(self, vid: str, trip_id: str, t: float, d: float, geom: TripGeom | None,
                 targets: list[int], stationary_sec: float) -> list[list[float]]:
        """Live feature rows (LIVE_COLS order, MISSING for unknowns) for this
        vehicle at shape distance *d* on *trip_id*, one per target stop index
        (into geom's sorted stops)."""
        v = self.veh.get(vid)
        nt = len(targets)
        out = [[_NAN] * N_LIVE for _ in range(nt)]
        h, wk = self._hw(t)

        # ---- vehicle-level (same for every target) -----------------------
        common = [_NAN] * N_LIVE
        common[_COL["live_path_cov"]] = 0.0
        common[_COL["hist_path_cov"]] = 0.0
        common[_COL["lap_path_cov"]] = 0.0
        common[_COL["veh_hist_n60"]] = 0.0
        common[_COL["dwell_n"]] = 0.0
        own180 = _NAN
        if v is not None:
            # own windowed speed (same trip)
            pos = v.pos
            for W, col in ((60, "own_speed_60"), (180, "own_speed_180"), (300, "own_speed_300")):
                # newest fix on this same trip at least W s old, at most W + 60 s
                hi_t = t - W
                best = None
                for e in reversed(pos):
                    if e[0] <= hi_t and e[1] == trip_id:
                        if e[0] >= hi_t - 60.0:
                            best = e
                        break
                if best is not None:
                    adv = d - best[2]
                    el = t - best[0]
                    if adv > -30.0 and el > 0:
                        common[_COL[col]] = max(adv, 0.0) / el
            own180 = common[_COL["own_speed_180"]]
            # feed ring
            feed = v.feed
            if feed:
                cur = feed[-1]
                if cur[0] == t:
                    vts, spd = cur[1], cur[2]
                    if vts == vts:
                        common[_COL["pos_age_s"]] = max(t - vts, 0.0)
                    common[_COL["fs_now"]] = spd
                ages = []
                s60 = []
                z600 = 0
                n600 = 0
                for e in reversed(feed):
                    if e[0] <= t - 600:
                        break
                    if e[0] > t:
                        continue
                    n600 += 1
                    if e[2] <= 0:
                        z600 += 1
                    if e[1] == e[1]:
                        ages.append(max(e[0] - e[1], 0.0))
                    if e[0] > t - 60 and e[2] == e[2]:
                        s60.append(e[2])
                if ages:
                    ages.sort()
                    m = len(ages)
                    common[_COL["pos_age_med600"]] = ages[m // 2] if m % 2 else 0.5 * (ages[m // 2 - 1] + ages[m // 2])
                if s60:
                    common[_COL["fs_mean60"]] = sum(s60) / len(s60)
                if n600:
                    common[_COL["fs_zero_frac600"]] = z600 / n600
                if cur[0] == t:
                    for W, col in ((60, "odo_spd_60"), (180, "odo_spd_180"), (300, "odo_m_300"),
                                   (600, "odo_spd_600"), (1200, "odo_spd_1200")):
                        hi_t = t - W
                        lo_t = hi_t - W
                        prev = None
                        for e in reversed(feed):
                            if e[0] <= hi_t:
                                if e[0] >= lo_t:
                                    prev = e
                                break
                        if prev is None:
                            continue
                        dodo = cur[3] - prev[3]
                        if not (dodo >= 0):
                            continue
                        if W == 300:
                            common[_COL[col]] = dodo
                        else:
                            dv = cur[1] - prev[1]
                            if dv > 0:
                                common[_COL[col]] = dodo / dv
            # vehicle-level own / historical ratio over the last hour
            sa = sb = 0.0
            sn = 0
            found = False
            lo_t = t - VEH_WINDOW_SEC
            for tc, ta, lt, hl in v.vlinks:
                if ta < t and lo_t < tc <= t:
                    found = True
                    if hl == hl:
                        sa += lt
                        sb += hl
                        sn += 1
            if found:
                common[_COL["veh_hist_n60"]] = float(sn)
                if sb > 0:
                    common[_COL["veh_hist_ratio60"]] = sa / sb
        age_s = common[_COL["pos_age_s"]]
        if age_s == age_s and own180 == own180 and own180 >= 0:
            common[_COL["pos_age_dist"]] = age_s * own180

        # ---- dwell at the current location -------------------------------
        if geom is not None and geom.dists:
            arr = self.tables.dwell.get(loc_key(geom, d))
            if arr is not None and len(arr):
                s = stationary_sec if stationary_sec == stationary_sec else 0.0
                n_all = len(arr)
                lo = int(np.searchsorted(arr, s, side="right"))
                m = n_all - lo
                common[_COL["dwell_n"]] = float(m)
                common[_COL["dwell_long_share"]] = (n_all - int(np.searchsorted(arr, 120.0, side="right"))) / n_all
                if m >= DWELL_MIN_N:
                    common[_COL["dwell_rem_med"]] = float(arr[lo + (m - 1) // 2]) - s
                    common[_COL["dwell_rem_p75"]] = float(arr[lo + (3 * (m - 1)) // 4]) - s
                    lo2 = int(np.searchsorted(arr, s + 120.0, side="right"))
                    common[_COL["dwell_p_more120"]] = (n_all - lo2) / m

        for r in out:
            r[:] = common

        # ---- path from the vehicle to each target -------------------------
        if geom is None or nt == 0:
            return _fill_missing(out)
        dists = geom.dists
        a = bisect_right(dists, d)
        b_max = max(targets)
        if a < 1 or b_max < a:
            return _fill_missing(out)
        span = dists[a] - dists[a - 1]
        w0 = (dists[a] - d) / span if span > 0 else 1.0
        w0 = min(max(w0, 0.0), 1.0)
        tables = self.tables
        own = v.own if v is not None else {}
        # running sums over links a..j
        wt = wf = wlt = wlt3 = wlt5 = 0.0
        nage = 0
        sage = 0.0
        whf = whl = whd = 0.0
        lf = la = 0.0
        nlage = 0
        slage = 0.0
        bf = bb = 0.0
        acc = {}
        cur_live = cur_hist = _NAN
        for j in range(a, b_max + 1):
            key = geom.keys[j]
            w = w0 if j == a else 1.0
            wt += w
            live = self._link_live(key, t)
            hist = tables.hist(key, h)
            histd = tables.hist_d(key, wk, h, hist)
            if live is not None:
                lt, m3, m5, age = live
                wf += w
                wlt += w * lt
                wlt3 += w * m3
                wlt5 += w * m5
                nage += 1
                sage += age
            if hist == hist:
                whf += w
                whl += w * hist
                whd += w * histd
            lap = own.get(key)
            llt = _NAN
            if lap is not None and lap[1] < t and lap[0] <= t and t - lap[0] <= LAP_WINDOW_SEC:
                llt = lap[2]
                lf += w
                la += w * llt
                nlage += 1
                slage += t - lap[0]
            fill = live[2] if live is not None else (llt if llt == llt else histd)
            if fill == fill:
                bf += w
                bb += w * fill
            if j == a:
                cur_live = live[2] if live is not None else _NAN
                cur_hist = histd
            acc[j] = (wt, wf, wlt, wlt3, wlt5, nage, sage, whf, whl, whd, lf, la, nlage, slage, bf, bb)
        C = _COL
        for r, b in zip(out, targets):
            if b < a:
                continue
            (wt, wf, wlt, wlt3, wlt5, nage, sage, whf, whl, whd, lf, la, nlage, slage, bf, bb) = acc[b]
            plen = dists[b] - d
            if wf > 0:
                lps = wlt / wf * wt
                r[C["live_path_sec_m3"]] = wlt3 / wf * wt
                r[C["live_path_sec_m5"]] = wlt5 / wf * wt
                if lps > 0:
                    r[C["live_speed_mps"]] = plen / lps
            if wt > 0:
                r[C["live_path_cov"]] = wf / wt
                r[C["hist_path_cov"]] = whf / wt
                r[C["lap_path_cov"]] = lf / wt
            if nage:
                r[C["live_path_age"]] = sage / nage
            if whf > 0:
                r[C["hist_path_sec"]] = whl / whf * wt
                r[C["hist_path_sec_dt"]] = whd / whf * wt
            if lf > 0:
                r[C["lap_path_sec"]] = la / lf * wt
            if nlage:
                r[C["lap_path_age"]] = slage / nlage
            if bf > 0:
                r[C["lap_fill_path_sec"]] = bb / bf * wt
            r[C["cur_link_live"]] = cur_live
            r[C["cur_link_hist"]] = cur_hist
            r[C["cur_link_frac"]] = w0
        return _fill_missing(out)

    # ── persistence ───────────────────────────────────────────────────────

    def prune(self, now: float) -> None:
        """Drop state no window can reach any more (call now and then)."""
        for key in [k for k, lst in self.links.items() if not lst or now - lst[-1][0] > LINK_WINDOW_SEC + 600]:
            del self.links[key]
        dead = []
        for vid, v in self.veh.items():
            lim = now - LAP_WINDOW_SEC
            for key in [k for k, x in v.own.items() if x[0] < lim]:
                del v.own[key]
            while v.vlinks and v.vlinks[0][0] < now - VEH_WINDOW_SEC - 120:
                v.vlinks.popleft()
            while v.feed and v.feed[0][0] < now - FEED_KEEP_SEC:
                v.feed.popleft()
            while v.pos and v.pos[0][0] < now - POS_KEEP_SEC:
                v.pos.popleft()
            if not v.own and not v.vlinks and not v.feed and not v.pos:
                dead.append(vid)
        for vid in dead:
            del self.veh[vid]

    def dump(self) -> dict:
        """Plain-Python snapshot of the state (tables excluded)."""
        return {
            "v": 1,
            "links": self.links,
            "veh": {
                vid: (list(v.feed), list(v.pos), v.trip, v.furthest, v.next_idx, v.last_t, v.last_d,
                      v.cross_idx, v.cross_t, v.streak, v.own, list(v.vlinks), v.last_feed_t, v.last_pos_t)
                for vid, v in self.veh.items()
            },
        }

    def load(self, blob: dict, geom) -> None:
        if not blob or blob.get("v") != 1:
            return
        self.links = {k: [tuple(x) for x in lst] for k, lst in blob.get("links", {}).items()}
        self.veh = {}
        for vid, s in blob.get("veh", {}).items():
            v = _Veh()
            (feed, pos, v.trip, v.furthest, v.next_idx, v.last_t, v.last_d,
             v.cross_idx, v.cross_t, v.streak, own, vlinks, v.last_feed_t, v.last_pos_t) = s
            v.feed = deque(tuple(x) for x in feed)
            v.pos = deque(tuple(x) for x in pos)
            v.own = {k: tuple(x) for k, x in own.items()}
            v.vlinks = deque(tuple(x) for x in vlinks)
            v.geom = geom(v.trip) if v.trip is not None else None
            if v.trip is not None and v.geom is None:
                v.trip = None
            self.veh[vid] = v


def _fill_missing(rows: list[list[float]]) -> list[list[float]]:
    for r in rows:
        for i, x in enumerate(r):
            if x != x:
                r[i] = MISSING
    return rows


# ---------------------------------------------------------------------------
# Side tables (written by scripts/run_pipeline.py)
# ---------------------------------------------------------------------------

def side_paths(train_dir: Path, day: str) -> tuple[Path, Path]:
    """(feed table, positions table) written next to a day's training rows."""
    d = Path(train_dir) / SIDE_DIR
    return d / f"{day}.feed.parquet", d / f"{day}.pos.parquet"


def derived_paths(train_dir: Path, day: str) -> tuple[Path, Path]:
    """(link traversals, stationary runs) derived from a day's side tables."""
    d = Path(train_dir) / SIDE_DIR
    return d / f"{day}.links.parquet", d / f"{day}.runs.parquet"


def _epoch_s(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts, utc=True)
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def write_side_tables(train_dir: Path, day: str, feed: pd.DataFrame,
                      positions: list[pd.DataFrame]) -> None:
    """Persist the two full-day streams the live features are rebuilt from.

    feed      — every vehicle entity of every snapshot, before trip inference:
                vehicle_id, t (feed header, epoch s), vehicle_ts, speed (m/s),
                odometer (km).
    positions — every on-route projected snapshot (labeling's projection):
                vehicle_id, trip_id, t, dist_along_m, run.
    """
    feed_path, pos_path = side_paths(train_dir, day)
    feed_path.parent.mkdir(parents=True, exist_ok=True)
    f = pd.DataFrame({
        "vehicle_id": feed["vehicle_id"].astype(str).to_numpy(),
        "t": _epoch_s(feed["timestamp"]),
        "vehicle_ts": pd.to_numeric(feed["vehicle_ts"], errors="coerce").to_numpy(dtype=float)
        if "vehicle_ts" in feed else np.nan,
        "speed": pd.to_numeric(feed["speed"], errors="coerce").to_numpy(dtype=float),
        "odometer": pd.to_numeric(feed["odometer"], errors="coerce").to_numpy(dtype=float)
        if "odometer" in feed else np.nan,
    })
    f = f.drop_duplicates(["vehicle_id", "t"]).sort_values(["t", "vehicle_id"])
    f.to_parquet(feed_path, index=False)
    if positions:
        p = pd.concat(positions, ignore_index=True)
    else:
        p = pd.DataFrame({"vehicle_id": [], "trip_id": [], "t": [], "dist_along_m": [], "run": []})
    p["vehicle_id"] = p["vehicle_id"].astype(str)
    p["trip_id"] = p["trip_id"].astype(str)
    p = p.sort_values(["t", "vehicle_id"]).reset_index(drop=True)
    p.to_parquet(pos_path, index=False)


# ---------------------------------------------------------------------------
# Replay (training)
# ---------------------------------------------------------------------------

WARMUP_SEC = LAP_WINDOW_SEC   # replay this much of the previous day first


def _load_streams(train_dir: Path, day: str, warmup: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datetime import date, timedelta
    feed_p, pos_p = side_paths(train_dir, day)
    feed = pd.read_parquet(feed_p)
    pos = pd.read_parquet(pos_p)
    if warmup:
        prev = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        pf, pp = side_paths(train_dir, prev)
        if pf.exists() and pp.exists() and len(feed):
            t0 = float(feed["t"].min())
            f0 = pd.read_parquet(pf)
            p0 = pd.read_parquet(pp)
            feed = pd.concat([f0[f0["t"] >= t0 - WARMUP_SEC], feed], ignore_index=True)
            pos = pd.concat([p0[p0["t"] >= t0 - WARMUP_SEC], pos], ignore_index=True)
    return feed, pos


def replay_day(train_dir: Path, day: str, geom, tables: LiveTables | None,
               queries: pd.DataFrame | None = None, collect_links: bool = False,
               warmup: bool = True):
    """Feed a day's side tables through a fresh LiveState in time order.

    queries (optional): rows with vehicle_id, trip_id, t, dist_along_m,
    stop_dist_along_m, stationary_sec — answered at their snapshot after every
    observation of that snapshot is in (the same state the daemon has when it
    reaches that vehicle, up to the t_avail < t rule). Returns
    (features array [len(queries), N_LIVE] or None, links DataFrame or None).
    """
    feed, pos = _load_streams(train_dir, day, warmup)
    links_out: list | None = [] if collect_links else None
    st = LiveState(tables, links_out=links_out)

    ft = feed["t"].to_numpy(dtype=float)
    fv = feed["vehicle_id"].astype(str).to_numpy(dtype=object)
    fts = feed["vehicle_ts"].to_numpy(dtype=float)
    fsp = feed["speed"].to_numpy(dtype=float)
    fod = feed["odometer"].to_numpy(dtype=float)
    fo = np.argsort(ft, kind="stable")

    pt = pos["t"].to_numpy(dtype=float)
    pv = pos["vehicle_id"].astype(str).to_numpy(dtype=object)
    ptr = pos["trip_id"].astype(str).to_numpy(dtype=object)
    pd_ = pos["dist_along_m"].to_numpy(dtype=float)
    po = np.argsort(pt, kind="stable")

    out = None
    if queries is not None and len(queries):
        qt = queries["t"].to_numpy(dtype=float)
        qv = queries["vehicle_id"].astype(str).to_numpy(dtype=object)
        qtr = queries["trip_id"].astype(str).to_numpy(dtype=object)
        qd = queries["dist_along_m"].to_numpy(dtype=float)
        qsd = queries["stop_dist_along_m"].to_numpy(dtype=float)
        qst = queries["stationary_sec"].to_numpy(dtype=float) if "stationary_sec" in queries else np.zeros(len(queries))
        # one query per (vehicle, t): all of its target stops at once
        order = np.lexsort((qv, qt))
        out = np.full((len(queries), N_LIVE), MISSING)
    else:
        qt = np.empty(0)
        order = np.empty(0, dtype=np.int64)

    times = np.unique(np.concatenate([ft, pt, qt]))
    i_f = i_p = i_q = 0
    nf, np_, nq = len(fo), len(po), len(order)
    for t in times:
        while i_f < nf and ft[fo[i_f]] <= t:
            k = fo[i_f]
            st.observe_feed(fv[k], ft[k], fts[k], fsp[k], fod[k])
            i_f += 1
        while i_p < np_ and pt[po[i_p]] <= t:
            k = po[i_p]
            st.observe_position(pv[k], ptr[k], pt[k], pd_[k], geom(ptr[k]))
            i_p += 1
        while i_q < nq and qt[order[i_q]] <= t:
            # gather this (vehicle, t) group
            k0 = order[i_q]
            j = i_q
            grp = []
            while j < nq and qt[order[j]] == qt[k0] and qv[order[j]] == qv[k0]:
                grp.append(order[j])
                j += 1
            i_q = j
            trip = qtr[k0]
            g = geom(trip)
            if g is None:
                continue
            dists = g.dists
            tgt = [_nearest(dists, qsd[k]) for k in grp]
            rows = st.features(qv[k0], trip, float(qt[k0]), float(qd[k0]), g, tgt, float(qst[k0]))
            for k, r in zip(grp, rows):
                out[k] = r
    links = None
    if collect_links:
        links = pd.DataFrame(links_out, columns=["key", "t", "lt", "vehicle_id"])
        if len(feed):
            day_start = pd.Timestamp(day, tz="UTC").timestamp()
            links = links[links["t"] >= day_start].reset_index(drop=True)
    return out, links


def _nearest(dists: list[float], x: float) -> int:
    i = bisect_right(dists, x)
    if i <= 0:
        return 0
    if i >= len(dists):
        return len(dists) - 1
    return i - 1 if abs(x - dists[i - 1]) <= abs(dists[i] - x) else i
