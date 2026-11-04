"""Unit tests for the data-layer event detection and configuration loader."""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.load import detect_events


def _promoted(rows: list[tuple[int, int, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["store_nbr", "item_nbr", "date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_detect_events_merges_gap_within_tolerance() -> None:
    """A single missing day inside a run is merged when gap_tolerance = 1."""
    promoted = _promoted([
        (1, 100, "2015-01-01"),
        (1, 100, "2015-01-02"),
        (1, 100, "2015-01-04"),  # 1-day gap -> same event
    ])
    events = detect_events(promoted, gap_tolerance=1)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["promo_start"] == pd.Timestamp("2015-01-01")
    assert row["promo_end"] == pd.Timestamp("2015-01-04")
    assert row["promo_len"] == 4  # calendar span
    assert row["n_promo_days"] == 3  # flagged days (flicker: span > days)


def test_detect_events_splits_beyond_tolerance() -> None:
    """A gap larger than the tolerance starts a new event."""
    promoted = _promoted([
        (1, 100, "2015-01-01"),
        (1, 100, "2015-01-02"),
        (1, 100, "2015-01-10"),  # 8-day gap -> new event
        (1, 100, "2015-01-11"),
    ])
    events = detect_events(promoted, gap_tolerance=1)
    assert len(events) == 2
    assert events["promo_len"].tolist() == [2, 2]


def test_detect_events_separate_item_stores() -> None:
    """Runs are detected independently per (store, item)."""
    promoted = _promoted([
        (1, 100, "2015-01-01"),
        (2, 100, "2015-01-01"),
        (1, 200, "2015-01-01"),
    ])
    events = detect_events(promoted, gap_tolerance=1)
    assert len(events) == 3
    assert (events["n_promo_days"] == 1).all()


def test_detect_events_empty() -> None:
    """An empty input yields an empty, well-formed frame."""
    events = detect_events(_promoted([]), gap_tolerance=1)
    assert events.empty
    assert "promo_len" in events.columns


def test_config_loads_and_resolves_paths() -> None:
    """The config loader returns typed values and absolute paths."""
    cfg = load_config()
    assert cfg.windows.pre_days == 90
    assert cfg.windows.post_days == 28
    assert cfg.selection.min_promo_days >= 1
    assert cfg.paths.data_raw.is_absolute()
    assert cfg.train_path().name == "train.csv"
