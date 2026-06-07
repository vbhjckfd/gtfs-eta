"""Load, index, and cache the GTFS static feed."""

from __future__ import annotations

import io
import os
import pickle
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from dotenv import load_dotenv

from src.utm import project_xy as _project_xy

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "gtfs_static"
CACHE_FILE = DATA_DIR / "_cache.pkl"
GTFS_STATIC_URL = os.environ.get(
    "GTFS_STATIC_URL", "https://track.ua-gis.com/gtfs/lviv/static.zip"
)


class StopInfo(NamedTuple):
    stop_id: str
    stop_name: str
    lat: float
    lon: float
    x: float  # EPSG:32635 easting
    y: float  # EPSG:32635 northing


class StopTime(NamedTuple):
    stop_id: str
    stop_sequence: int
    arrival_time: str   # HH:MM:SS (may exceed 24h for overnight)
    departure_time: str
    shape_dist_traveled: float | None


class TripInfo(NamedTuple):
    trip_id: str
    route_id: str
    service_id: str
    shape_id: str
    direction_id: int
    stop_times: list[StopTime]


class GTFSStatic:
    def __init__(self) -> None:
        self._routes: pd.DataFrame | None = None
        self._trips: pd.DataFrame | None = None
        self._stops: dict[str, StopInfo] = {}
        self._stop_times: pd.DataFrame | None = None
        self._shapes: dict[str, LineString] = {}        # shape_id → projected LineString
        self._shape_lengths: dict[str, float] = {}
        self._stop_distances: dict[tuple[str, str], float] = {}  # (shape_id, stop_id) → dist along
        self._calendar: pd.DataFrame | None = None
        self._calendar_dates: pd.DataFrame | None = None
        self._calendar_parsed: pd.DataFrame | None = None   # pre-parsed dates
        self._active_services_cache: dict[date, set[str]] = {}
        self._trip_index: dict[str, TripInfo] = {}
        self._route_trips: dict[str, list[str]] = {}   # route_id → [trip_id]
        self.feed_tz: ZoneInfo = ZoneInfo("Europe/Kiev")  # overwritten during parse

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, force_download: bool = False, force_rebuild: bool = False) -> "GTFSStatic":
        """Download (if needed), extract, parse, and cache the static feed."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATA_DIR / "static.zip"

        if force_download or not zip_path.exists():
            _download(GTFS_STATIC_URL, zip_path)

        if not force_rebuild and CACHE_FILE.exists():
            self._load_cache()
        else:
            self._extract(zip_path)
            self._parse()
            self._save_cache()

        return self

    def get_trip(self, trip_id: str) -> TripInfo | None:
        return self._trip_index.get(trip_id)

    def get_shape_linestring(self, shape_id: str) -> LineString | None:
        return self._shapes.get(shape_id)

    def get_shape_length(self, shape_id: str) -> float:
        return self._shape_lengths.get(shape_id, 0.0)

    def get_stop(self, stop_id: str) -> StopInfo | None:
        return self._stops.get(str(stop_id))

    def get_stop_distance_along_shape(self, shape_id: str, stop_id: str) -> float | None:
        """Distance in metres along the projected shape to the nearest point to this stop."""
        return self._stop_distances.get((shape_id, str(stop_id)))

    def active_trips_for_route(self, route_id: str, now: datetime) -> list[str]:
        """Return trip_ids that are scheduled to run at *now* on the given route."""
        trip_ids = self._route_trips.get(str(route_id), [])
        active_services = self._active_services(now.date())
        result = []
        for tid in trip_ids:
            info = self._trip_index.get(tid)
            if info and info.service_id in active_services:
                if self._trip_running_at(info, now):
                    result.append(tid)
        return result

    def all_routes(self) -> pd.DataFrame:
        return self._routes.copy()

    def all_trips(self) -> pd.DataFrame:
        return self._trips.copy()

    def stop_times_for_trip(self, trip_id: str) -> pd.DataFrame:
        return self._stop_times[self._stop_times["trip_id"] == trip_id].copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, zip_path: Path) -> None:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)

    def _parse(self) -> None:
        self._routes = pd.read_csv(DATA_DIR / "routes.txt", dtype=str)
        self._trips = pd.read_csv(DATA_DIR / "trips.txt", dtype=str)

        agency_path = DATA_DIR / "agency.txt"
        if agency_path.exists():
            agency = pd.read_csv(agency_path, dtype=str)
            if "agency_timezone" in agency.columns and len(agency) > 0:
                tz_name = agency["agency_timezone"].iloc[0]
                try:
                    self.feed_tz = ZoneInfo(tz_name)
                except Exception:
                    pass
        stops_raw = pd.read_csv(DATA_DIR / "stops.txt", dtype=str)
        self._stop_times = pd.read_csv(DATA_DIR / "stop_times.txt", dtype=str)
        shapes_raw = pd.read_csv(DATA_DIR / "shapes.txt", dtype=str)

        cal_path = DATA_DIR / "calendar.txt"
        caldates_path = DATA_DIR / "calendar_dates.txt"
        self._calendar = pd.read_csv(cal_path, dtype=str) if cal_path.exists() else pd.DataFrame()
        self._calendar_dates = (
            pd.read_csv(caldates_path, dtype=str) if caldates_path.exists() else pd.DataFrame()
        )
        if not self._calendar.empty:
            self._calendar_parsed = self._calendar.copy()
            self._calendar_parsed["start_date"] = pd.to_datetime(
                self._calendar_parsed["start_date"], format="%Y%m%d"
            )
            self._calendar_parsed["end_date"] = pd.to_datetime(
                self._calendar_parsed["end_date"], format="%Y%m%d"
            )
        else:
            self._calendar_parsed = pd.DataFrame()

        self._build_stops(stops_raw)
        self._build_shapes(shapes_raw)
        self._build_trip_index()
        self._build_stop_distances()
        self._build_route_trips()

    def _build_stops(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            lat, lon = float(row["stop_lat"]), float(row["stop_lon"])
            x, y = _project_xy(lon, lat)
            self._stops[str(row["stop_id"])] = StopInfo(
                stop_id=str(row["stop_id"]),
                stop_name=row.get("stop_name", ""),
                lat=lat,
                lon=lon,
                x=x,
                y=y,
            )

    def _build_shapes(self, df: pd.DataFrame) -> None:
        df["shape_pt_lat"] = df["shape_pt_lat"].astype(float)
        df["shape_pt_lon"] = df["shape_pt_lon"].astype(float)
        df["shape_pt_sequence"] = df["shape_pt_sequence"].astype(int)

        for shape_id, grp in df.groupby("shape_id"):
            grp = grp.sort_values("shape_pt_sequence")
            coords = [
                _project_xy(row["shape_pt_lon"], row["shape_pt_lat"])
                for _, row in grp.iterrows()
            ]
            if len(coords) >= 2:
                ls = LineString(coords)
                self._shapes[str(shape_id)] = ls
                self._shape_lengths[str(shape_id)] = ls.length

    def _build_trip_index(self) -> None:
        st = self._stop_times.copy()
        if "shape_dist_traveled" not in st.columns:
            st["shape_dist_traveled"] = None

        # Convert stop_sequence to int for sorting
        st["stop_sequence"] = st["stop_sequence"].astype(int)
        st = st.sort_values(["trip_id", "stop_sequence"])

        grouped = st.groupby("trip_id")
        trip_map = self._trips.set_index("trip_id").to_dict("index")

        for trip_id, grp in grouped:
            meta = trip_map.get(str(trip_id), {})
            stop_times_list = [
                StopTime(
                    stop_id=str(r["stop_id"]),
                    stop_sequence=int(r["stop_sequence"]),
                    arrival_time=str(r.get("arrival_time", "")),
                    departure_time=str(r.get("departure_time", "")),
                    shape_dist_traveled=(
                        float(r["shape_dist_traveled"])
                        if r.get("shape_dist_traveled") not in (None, "", "nan")
                        else None
                    ),
                )
                for _, r in grp.iterrows()
            ]
            self._trip_index[str(trip_id)] = TripInfo(
                trip_id=str(trip_id),
                route_id=str(meta.get("route_id", "")),
                service_id=str(meta.get("service_id", "")),
                shape_id=str(meta.get("shape_id", "")),
                direction_id=int(meta.get("direction_id", 0) or 0),
                stop_times=stop_times_list,
            )

    def _build_stop_distances(self) -> None:
        """For every (shape_id, stop_id) pair referenced by a trip, compute distance along shape."""
        for trip_id, info in self._trip_index.items():
            shape = self._shapes.get(info.shape_id)
            if shape is None:
                continue
            for st in info.stop_times:
                key = (info.shape_id, st.stop_id)
                if key in self._stop_distances:
                    continue
                stop = self._stops.get(st.stop_id)
                if stop is None:
                    continue
                pt = Point(stop.x, stop.y)
                dist = shape.project(pt)
                self._stop_distances[key] = dist

    def _build_route_trips(self) -> None:
        for trip_id, info in self._trip_index.items():
            self._route_trips.setdefault(info.route_id, []).append(trip_id)

    def _active_services(self, d: date) -> set[str]:
        if d in self._active_services_cache:
            return self._active_services_cache[d]

        active: set[str] = set()
        dow = d.strftime("%A").lower()  # monday, tuesday, ...
        date_str = d.strftime("%Y%m%d")

        if not self._calendar.empty:
            cal = self._calendar_parsed
            mask = (
                (cal["start_date"] <= pd.Timestamp(d))
                & (cal["end_date"] >= pd.Timestamp(d))
                & (cal[dow] == "1")
            )
            active.update(cal.loc[mask, "service_id"].tolist())

        if not self._calendar_dates.empty:
            cdates = self._calendar_dates
            added = cdates[(cdates["date"] == date_str) & (cdates["exception_type"] == "1")]
            active.update(added["service_id"].tolist())
            removed = cdates[(cdates["date"] == date_str) & (cdates["exception_type"] == "2")]
            active -= set(removed["service_id"].tolist())

        self._active_services_cache[d] = active
        return active

    def _trip_running_at(self, info: TripInfo, now: datetime) -> bool:
        """Rough check: is `now` between first departure and last arrival of this trip?"""
        if not info.stop_times:
            return False
        # Use date in local wall-clock time (strip tz for naive GTFS times)
        now_naive = now.replace(tzinfo=None) if now.tzinfo is not None else now
        now_date = now_naive.date()
        first_dep = _parse_gtfs_time(info.stop_times[0].departure_time, now_date)
        last_arr = _parse_gtfs_time(info.stop_times[-1].arrival_time, now_date)
        if first_dep is None or last_arr is None:
            return True  # can't determine, include
        # Add a small buffer so vehicles that started slightly early/late are included
        return (first_dep - timedelta(minutes=10)) <= now_naive <= (last_arr + timedelta(minutes=30))

    def _save_cache(self) -> None:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(
                {
                    "routes": self._routes,
                    "trips": self._trips,
                    "stops": self._stops,
                    "stop_times": self._stop_times,
                    "shapes": self._shapes,
                    "shape_lengths": self._shape_lengths,
                    "stop_distances": self._stop_distances,
                    "calendar": self._calendar,
                    "calendar_parsed": self._calendar_parsed,
                    "calendar_dates": self._calendar_dates,
                    "trip_index": self._trip_index,
                    "route_trips": self._route_trips,
                    "feed_tz": self.feed_tz,
                },
                f,
            )

    def _load_cache(self) -> None:
        with open(CACHE_FILE, "rb") as f:
            d = pickle.load(f)
        self._routes = d["routes"]
        self._trips = d["trips"]
        self._stops = d["stops"]
        self._stop_times = d["stop_times"]
        self._shapes = d["shapes"]
        self._shape_lengths = d["shape_lengths"]
        self._stop_distances = d["stop_distances"]
        self._calendar = d["calendar"]
        self._calendar_parsed = d.get("calendar_parsed", pd.DataFrame())
        self._calendar_dates = d["calendar_dates"]
        self._trip_index = d["trip_index"]
        self._route_trips = d["route_trips"]
        self.feed_tz = d.get("feed_tz", ZoneInfo("Europe/Kiev"))
        self._active_services_cache = {}


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url} → {dest}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"  saved {dest.stat().st_size / 1e6:.1f} MB")


def _parse_gtfs_time(time_str: str, base_date: date) -> datetime | None:
    """Parse GTFS HH:MM:SS (may be ≥24:00:00 for overnight) into a naive local datetime."""
    if not time_str or time_str == "nan":
        return None
    parts = time_str.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return datetime(base_date.year, base_date.month, base_date.day) + timedelta(
            hours=h, minutes=m, seconds=s
        )
    except ValueError:
        return None


def _parse_gtfs_time_utc(
    time_str: str, base_date: date, feed_tz: ZoneInfo
) -> datetime | None:
    """Parse GTFS time and return a UTC-aware datetime."""
    local = _parse_gtfs_time(time_str, base_date)
    if local is None:
        return None
    return local.replace(tzinfo=feed_tz).astimezone(timezone.utc)


# Module-level singleton
_instance: GTFSStatic | None = None


def get_gtfs(force_download: bool = False, force_rebuild: bool = False) -> GTFSStatic:
    global _instance
    if _instance is None:
        _instance = GTFSStatic().load(
            force_download=force_download, force_rebuild=force_rebuild
        )
    return _instance
