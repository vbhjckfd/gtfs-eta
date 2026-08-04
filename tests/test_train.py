import pandas as pd

from src.train import _time_split


def _rows(date_route_pairs):
    return pd.DataFrame(
        {"date": [d for d, _ in date_route_pairs], "route_id": [r for _, r in date_route_pairs]}
    )


class TestTimeSplitNewRoute:
    """A route whose entire history postdates the chronological cutoff must
    still reach train_df — otherwise no amount of rerunning teaches the model
    it exists (#route137/A67)."""

    def test_route_confined_to_test_window_is_carved_into_train(self):
        old_dates = [f"2026-07-{d:02d}" for d in range(1, 21)]  # 20 old days, established route
        new_dates = ["2026-08-01", "2026-08-02"]  # 2 new days, brand-new route

        rows = [(d, "OLD") for d in old_dates for _ in range(10)]
        rows += [(d, "NEW") for d in new_dates for _ in range(10)]
        df = _rows(rows)

        train_df, test_df = _time_split(df, test_fraction=0.2)

        assert (train_df["route_id"] == "NEW").sum() > 0
        assert (test_df["route_id"] == "NEW").sum() > 0
        # ~80% moved into train, per the carve-out fraction.
        n_new_total = (df["route_id"] == "NEW").sum()
        n_new_train = (train_df["route_id"] == "NEW").sum()
        assert n_new_train / n_new_total > 0.5

    def test_established_route_split_is_untouched(self):
        old_dates = [f"2026-07-{d:02d}" for d in range(1, 21)]
        new_dates = ["2026-08-01", "2026-08-02"]
        rows = [(d, "OLD") for d in old_dates for _ in range(10)]
        rows += [(d, "OLD") for d in new_dates for _ in range(10)]  # same route throughout
        df = _rows(rows)

        train_df, test_df = _time_split(df, test_fraction=0.2)

        # No route is new relative to the cutoff — split stays purely chronological.
        assert train_df["date"].max() < test_df["date"].min()
