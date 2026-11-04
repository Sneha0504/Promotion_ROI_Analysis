"""Shared feature helpers for the analysis panel and the counterfactual models.

This module centralizes the data-quality rules that every downstream model relies on,
so interrupted-time-series and synthetic-control code share exactly one implementation:

* the **full daily calendar** over an event window (Favorita stores no zero-sales rows,
  so an absent day is either genuine zero demand on an open day or a store closure);
* the **store-closure mask** derived from ``transactions.csv`` (a store-day with no
  transactions row is a closure, masked rather than treated as zero demand);
* the **holiday indicator** per (store, date) from ``holidays_events.csv`` honoring the
  national/regional/local locale and the ``transferred`` flag;
* the daily **oil price** (forward-filled), an optional interrupted-time-series regressor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config

# Holiday calendar rows whose dates genuinely shift demand. ``Work Day`` is a
# compensating *working* day (the opposite of a holiday) and is deliberately excluded.
_HOLIDAY_TYPES = frozenset({"Holiday", "Transfer", "Additional", "Bridge", "Event"})


def full_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return every calendar day in the inclusive ``[start, end]`` window."""
    return pd.date_range(start=start, end=end, freq="D")


def load_open_days(cfg: Config) -> dict[int, set[pd.Timestamp]]:
    """Map each store to the set of dates on which it recorded transactions.

    ``transactions.csv`` holds one row per store-day the store actually traded (there are
    no ``transactions = 0`` rows), so a (store, date) absent from this mapping is a
    closure. Downstream code masks such days instead of imputing zero demand.
    """
    txn = pd.read_csv(cfg.raw_file("transactions"), parse_dates=["date"])
    return {
        int(store): set(group["date"])
        for store, group in txn.groupby("store_nbr", sort=False)
    }


def closure_mask(open_days: dict[int, set[pd.Timestamp]], store_nbr: int,
                 dates: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array, ``True`` where the store was *closed* (no transactions) on a date."""
    store_open = open_days.get(int(store_nbr), set())
    return np.array([d not in store_open for d in dates], dtype=bool)


class HolidayCalendar:
    """Resolve a boolean holiday indicator for any (store, date) combination.

    The Favorita calendar is scoped by locale: ``National`` holidays apply to every
    store, ``Regional`` holidays to stores whose ``state`` matches ``locale_name``, and
    ``Local`` holidays to stores whose ``city`` matches. Rows flagged ``transferred`` are
    dropped (the holiday was moved; the observed date arrives as a separate ``Transfer``
    row), and ``Work Day`` rows are ignored (a working day, not a holiday).
    """

    def __init__(self, cfg: Config) -> None:
        holidays = pd.read_csv(cfg.raw_file("holidays"), parse_dates=["date"])
        holidays = holidays[holidays["type"].isin(_HOLIDAY_TYPES)]
        holidays = holidays[~holidays["transferred"].astype(bool)]

        national = holidays[holidays["locale"] == "National"]
        regional = holidays[holidays["locale"] == "Regional"]
        local = holidays[holidays["locale"] == "Local"]

        self._national: set[pd.Timestamp] = set(national["date"])
        self._regional: dict[str, set[pd.Timestamp]] = {
            name: set(grp["date"]) for name, grp in regional.groupby("locale_name")
        }
        self._local: dict[str, set[pd.Timestamp]] = {
            name: set(grp["date"]) for name, grp in local.groupby("locale_name")
        }

    def flags(self, dates: pd.DatetimeIndex, state: str, city: str) -> np.ndarray:
        """Boolean array, ``True`` where the given store's locale has a holiday."""
        applicable = set(self._national)
        applicable |= self._regional.get(state, set())
        applicable |= self._local.get(city, set())
        return np.array([d in applicable for d in dates], dtype=bool)


def load_oil(cfg: Config) -> pd.Series:
    """Return a daily, forward-filled WTI oil-price series indexed by date.

    ``oil.csv`` is quoted on business days only and carries occasional null prices, so it
    is reindexed to a dense daily calendar and both forward- then back-filled. Oil is only
    ever an optional regressor, never a filter.
    """
    oil = pd.read_csv(cfg.raw_file("oil"), parse_dates=["date"]).set_index("date")
    series = oil["dcoilwtico"].sort_index()
    full = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full).ffill().bfill()
