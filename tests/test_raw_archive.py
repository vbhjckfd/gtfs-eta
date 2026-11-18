"""Packing old raw/ days into one archive object, and reading them back.

R2 storage for raw/ is ~400 MB a day as ~8.6k small protobufs; one xz tar per
day is ~8x smaller. Readers must not notice which form a day is in.
"""

from __future__ import annotations

import io
import os

for _k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ.setdefault(_k, "test")

import pytest  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402
from google.transit import gtfs_realtime_pb2  # noqa: E402

from src import snapshots  # noqa: E402

DAY = "2026-09-01"


class FakeR2:
    def __init__(self, store: dict[str, bytes]):
        self.store = store

    def get_paginator(self, _op):
        store = self.store

        class _P:
            def paginate(self, Bucket, Prefix, Delimiter=None):
                keys = sorted(k for k in store if k.startswith(Prefix))
                if Delimiter is None:
                    yield {"Contents": [{"Key": k} for k in keys]}
                    return
                prefixes = sorted({Prefix + k[len(Prefix):].split(Delimiter)[0] + Delimiter
                                   for k in keys if Delimiter in k[len(Prefix):]})
                yield {"CommonPrefixes": [{"Prefix": p} for p in prefixes],
                       "Contents": [{"Key": k} for k in keys if Delimiter not in k[len(Prefix):]]}

        return _P()

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.store[Key])}

    def download_file(self, Bucket, Key, Filename):
        if Key not in self.store:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        with open(Filename, "wb") as f:
            f.write(self.store[Key])

    def upload_file(self, Filename, Bucket, Key):
        with open(Filename, "rb") as f:
            self.store[Key] = f.read()

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.store.pop(o["Key"], None)
        return {}


def _feed(sec: int) -> bytes:
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.header.gtfs_realtime_version = "2.0"
    msg.header.timestamp = 1756684800 + sec
    v = msg.entity.add(id="1").vehicle
    v.vehicle.id = "v1"
    v.position.latitude, v.position.longitude = 49.84, 24.03
    return msg.SerializeToString()


def _raw(secs) -> dict[str, bytes]:
    return {f"raw/{DAY}/{DAY}T00:00:{s:02d}Z.pb": _feed(s) for s in secs}


@pytest.fixture
def r2(tmp_path, monkeypatch):
    fake = FakeR2({})
    monkeypatch.setattr(snapshots, "_make_client", lambda: fake)
    monkeypatch.setattr(snapshots, "ARCHIVE_CACHE", tmp_path / "cache")
    return fake


def test_archived_day_reads_like_a_raw_day(r2):
    r2.store.update(_raw([0, 10, 20]))
    before = snapshots.load_snapshots_df(date_str=DAY)

    assert snapshots.archive_day(r2, DAY) == 3

    assert list(r2.store) == [f"archive/raw/{DAY}.tar.xz"]
    assert snapshots.list_snapshot_days() == [DAY]
    assert snapshots.load_snapshots_df(date_str=DAY).equals(before)
    assert len(before) == 3


def test_packing_keeps_what_an_interrupted_run_already_archived(r2):
    # A run that died after the upload but mid-delete leaves a full archive
    # and a partial raw/ day; re-packing must not shrink the archive to it.
    r2.store.update(_raw([0, 10, 20]))
    snapshots.archive_day(r2, DAY)
    r2.store.update(_raw([20, 30]))

    assert snapshots.archive_day(r2, DAY) == 4
    assert len(snapshots.list_snapshot_keys(date_str=DAY)) == 4


def test_raw_days_alone_skips_archived_ones(r2):
    r2.store.update(_raw([0]))
    snapshots.archive_day(r2, DAY)
    r2.store["raw/2026-09-02/2026-09-02T00:00:00Z.pb"] = _feed(0)

    assert snapshots.list_snapshot_days() == [DAY, "2026-09-02"]
    assert snapshots.list_snapshot_days(archived=False) == ["2026-09-02"]


def test_day_with_neither_form_is_empty(r2):
    assert snapshots.list_snapshot_keys(date_str=DAY) == []
