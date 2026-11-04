"""Data layer: promotion-event detection and sample selection.

This module is the entry point for Unit 1 of the pipeline. It uses DuckDB to run SQL
directly against the raw CSVs (``read_csv_auto``) so the 4.7 GB ``train.csv`` is scanned
as a stream and never loaded whole into memory. From the scan it:

1. profiles the raw data (headline quality counts);
2. detects promotion *events* — maximal runs of consecutive ``onpromotion = TRUE`` days
   per (store_nbr, item_nbr), merging gaps of <= ``promo_flag_gap_tolerance`` days;
3. applies the configured selection criteria (promo length, pre-period history and
   density, clean non-overlapping windows, store open across the windows);
4. caps the result to ``max_events`` deterministically by **pre-period-only** sales
   volume -- ``SUM(unit_sales)`` over each event's ``[pre_start, promo_start)`` window,
   which is strictly pre-intervention and therefore independent of the promotion
   outcome -- and writes it to ``data/processed/selected_events.parquet`` plus a CSV copy.

The run is idempotent (outputs are overwritten) and logs progress via ``logging``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd

from src.config import Config, load_config
from src.logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# DuckDB helpers
# --------------------------------------------------------------------------------------
def read_sql(cfg: Config, name: str) -> str:
    """Return the text of ``sql/<name>.sql``."""
    sql_path = cfg.paths.sql / f"{name}.sql"
    return sql_path.read_text(encoding="utf-8")


def connect_duckdb(
    threads: int = 4,
    memory_limit: str = "6GB",
    temp_dir: str | Path | None = None,
    max_temp_size: str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection bounded in threads, memory and disk spill.

    Scans are streaming, so a generous ``memory_limit`` keeps aggregation and result
    materialization in RAM. Disk spilling is the real risk on this machine (only a few GB
    free), so when ``temp_dir`` is given it is created and set as DuckDB's
    ``temp_directory`` (a project-local, cleanable path off the system temp volume), and
    ``max_temp_size`` caps how much DuckDB may spill there -- a query that would exceed
    the cap errors out instead of silently filling the disk.
    """
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    if temp_dir is not None:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    if max_temp_size is not None:
        con.execute(f"PRAGMA max_temp_directory_size='{max_temp_size}'")
    return con


def connect_from_config(cfg: Config) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection using the bounds configured in ``config.yaml``."""
    d = cfg.duckdb
    return connect_duckdb(d.threads, d.memory_limit, d.temp_dir, d.max_temp_size)


# --------------------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------------------
def profile_train(cfg: Config, con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """Run the one-pass profile of ``train.csv`` and return the headline counts."""
    logger.info("Profiling train.csv (single streaming scan)...")
    sql = read_sql(cfg, "profile_train")
    row = con.execute(sql, [str(cfg.train_path())]).df().iloc[0]
    result = row.to_dict()
    logger.info("Profile complete: %s rows, dates %s..%s",
                f"{int(result['n_rows']):,}", result["first_date"], result["last_date"])
    return result


# --------------------------------------------------------------------------------------
# Extraction (SQL)
# --------------------------------------------------------------------------------------
def load_candidate_series(cfg: Config, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per item-store aggregates (within the reliable-flag window) for selection."""
    logger.info("Scanning candidate series aggregates...")
    sql = read_sql(cfg, "candidate_series")
    df = con.execute(sql, [str(cfg.train_path()), cfg.reliable_flag_start]).df()
    df["first_date"] = pd.to_datetime(df["first_date"])
    df["last_date"] = pd.to_datetime(df["last_date"])
    df["nonzero_frac"] = df["n_nonzero_days"] / df["n_days"]
    logger.info("Candidate series with >=1 promoted day: %s", f"{len(df):,}")
    return df


def load_promoted_rows(cfg: Config, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """All promoted (store, item, date) rows within the reliable-flag window."""
    logger.info("Scanning promoted rows for event detection...")
    sql = read_sql(cfg, "promoted_rows")
    df = con.execute(sql, [str(cfg.train_path()), cfg.reliable_flag_start]).df()
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Promoted rows: %s", f"{len(df):,}")
    return df


def query_daily_panel(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    pairs: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Daily unit_sales panel for the given (store_nbr, item_nbr) pairs and date range.

    Provided for the later panel builder. ``pairs`` must have columns
    ``store_nbr`` and ``item_nbr``; it is registered as the ``target_pairs`` relation.
    """
    target_pairs = pairs[["store_nbr", "item_nbr"]].drop_duplicates()
    con.register("target_pairs", target_pairs)
    try:
        sql = read_sql(cfg, "daily_panel")
        df = con.execute(sql, [str(cfg.train_path()), str(start), str(end)]).df()
    finally:
        con.unregister("target_pairs")
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_pre_period_volume(
    cfg: Config, con: duckdb.DuckDBPyConnection, events_window: pd.DataFrame
) -> pd.DataFrame:
    """Sum ``unit_sales`` over each event's ``[pre_start, promo_start)`` pre-period.

    This is the outcome-independent ranking key for the cap: the window ends strictly
    before the promotion starts, so it encodes only pre-intervention demand, never the
    promotion or post-period response. Computed in DuckDB against ``train.csv``.

    Args:
        events_window: candidate events with columns ``evkey, store_nbr, item_nbr,
            pre_start, promo_start``.

    Returns:
        DataFrame with ``evkey`` and ``pre_period_unit_sales`` (one row per matched event).
    """
    target = events_window[["evkey", "store_nbr", "item_nbr", "pre_start", "promo_start"]]
    con.register("target_events", target)
    try:
        sql = read_sql(cfg, "pre_period_volume")
        vol = con.execute(sql, [str(cfg.train_path()), cfg.reliable_flag_start]).df()
    finally:
        con.unregister("target_events")
    return vol


# --------------------------------------------------------------------------------------
# Event detection
# --------------------------------------------------------------------------------------
def detect_events(promoted: pd.DataFrame, gap_tolerance: int) -> pd.DataFrame:
    """Collapse promoted days into maximal runs, merging gaps of <= ``gap_tolerance``.

    Args:
        promoted: rows with columns ``store_nbr, item_nbr, date`` (promoted days only).
        gap_tolerance: calendar-day gap tolerated inside one event. Two promoted days
            belong to the same event when their date difference is
            ``<= gap_tolerance + 1`` (a difference of 1 is strictly consecutive).

    Returns:
        One row per event with ``store_nbr, item_nbr, promo_start, promo_end,
        promo_len`` (calendar span in days) and ``n_promo_days`` (flagged days in run).
    """
    if promoted.empty:
        return pd.DataFrame(
            columns=["store_nbr", "item_nbr", "promo_start", "promo_end",
                     "promo_len", "n_promo_days"]
        )

    df = promoted[["store_nbr", "item_nbr", "date"]].drop_duplicates()
    df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

    grp = df.groupby(["store_nbr", "item_nbr"], sort=False)
    day_gap = grp["date"].diff().dt.days
    # New run when first row of a group (NaN gap) or gap exceeds the merge tolerance.
    new_run = day_gap.isna() | (day_gap > (gap_tolerance + 1))
    df["run_id"] = new_run.cumsum()

    events = (
        df.groupby(["store_nbr", "item_nbr", "run_id"], sort=False)
        .agg(promo_start=("date", "min"),
             promo_end=("date", "max"),
             n_promo_days=("date", "size"))
        .reset_index()
        .drop(columns="run_id")
    )
    events["promo_len"] = (events["promo_end"] - events["promo_start"]).dt.days + 1
    logger.info("Detected %s raw promotion events across %s item-stores",
                f"{len(events):,}", f"{df[['store_nbr', 'item_nbr']].drop_duplicates().shape[0]:,}")
    return events


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------
@dataclass
class SelectionStats:
    """Counts dropped at each selection stage, for the data-quality report."""

    raw_events: int = 0
    after_length: int = 0
    after_history: int = 0
    after_nonzero: int = 0
    after_clean_window: int = 0
    after_store_open: int = 0
    final: int = 0


def _store_operating_span(cfg: Config) -> pd.DataFrame:
    """Return per-store first/last transaction dates (store operational span)."""
    txn = pd.read_csv(cfg.raw_file("transactions"), parse_dates=["date"])
    span = txn.groupby("store_nbr")["date"].agg(store_first="min", store_last="max")
    return span.reset_index()


def apply_selection(
    events: pd.DataFrame,
    series: pd.DataFrame,
    items: pd.DataFrame,
    stores: pd.DataFrame,
    store_span: pd.DataFrame,
    cfg: Config,
    con: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, SelectionStats]:
    """Filter raw events to the analyzable sample and attach metadata.

    Applies, in order: promo-length bounds; sufficient pre-period history; sufficient
    non-zero sales density (series-level proxy within the reliable window); clean
    (non-overlapping) pre+promo+post windows for the same item-store; and store open
    across the windows. Finally caps to ``max_events`` by descending **pre-period-only**
    sales volume -- ``SUM(unit_sales)`` over each event's ``[pre_start, promo_start)``
    window (computed in DuckDB via ``con``) -- which is deterministic and strictly
    pre-intervention, so the sample is independent of the promotion outcome.
    """
    stats = SelectionStats(raw_events=len(events))
    sel = cfg.selection
    pre = pd.Timedelta(days=cfg.windows.pre_days)
    post = pd.Timedelta(days=cfg.windows.post_days)
    reliable_start = pd.Timestamp(cfg.reliable_flag_start)

    ev = events.copy()
    ev["pre_start"] = ev["promo_start"] - pre
    ev["post_end"] = ev["promo_end"] + post

    # 1. Promotion length within bounds.
    ev = ev[(ev["promo_len"] >= sel.min_promo_days)
            & (ev["promo_len"] <= sel.max_promo_days)]
    stats.after_length = len(ev)

    # 2. Enough pre-period history: the item-store must be observed from the reliable
    #    window start on/before pre_start, and pre_start must be within that window.
    ev = ev.merge(
        series[["store_nbr", "item_nbr", "first_date", "nonzero_frac"]],
        on=["store_nbr", "item_nbr"], how="left",
    )
    ev = ev[(ev["pre_start"] >= reliable_start) & (ev["first_date"] <= ev["pre_start"])]
    stats.after_history = len(ev)

    # 3. Sufficient non-zero sales density (series-level proxy for a modelable series).
    ev = ev[ev["nonzero_frac"] >= sel.min_pre_nonzero_frac]
    stats.after_nonzero = len(ev)

    # 4. Clean window: no OTHER event of the same item-store intersects [pre_start,
    #    post_end]. Computed from the full detected-event list per item-store.
    all_ev = events[["store_nbr", "item_nbr", "promo_start", "promo_end"]]
    keep_clean: list[bool] = []
    others_by_pair = {k: v for k, v in all_ev.groupby(["store_nbr", "item_nbr"], sort=False)}
    for row in ev.itertuples(index=False):
        others = others_by_pair.get((row.store_nbr, row.item_nbr))
        clean = True
        if others is not None and len(others) > 1:
            for o in others.itertuples(index=False):
                if o.promo_start == row.promo_start and o.promo_end == row.promo_end:
                    continue  # the event itself
                # overlap of [o.promo_start, o.promo_end] with [pre_start, post_end]
                if o.promo_start <= row.post_end and o.promo_end >= row.pre_start:
                    clean = False
                    break
        keep_clean.append(clean)
    ev = ev[pd.Series(keep_clean, index=ev.index)]
    stats.after_clean_window = len(ev)

    # 5. Store open across the windows (operational span covers pre_start..post_end).
    if sel.require_store_open:
        ev = ev.merge(store_span, on="store_nbr", how="left")
        ev = ev[(ev["store_first"] <= ev["pre_start"])
                & (ev["store_last"] >= ev["post_end"])]
    stats.after_store_open = len(ev)

    # Pre-period-only volume for the cap ranking: SUM(unit_sales) over [pre_start,
    # promo_start). Strictly pre-intervention, so the ranking cannot leak the outcome.
    ev = ev.reset_index(drop=True)
    ev["evkey"] = range(len(ev))
    vol = compute_pre_period_volume(cfg, con, ev)
    ev = ev.merge(vol, on="evkey", how="left")
    ev["pre_period_unit_sales"] = ev["pre_period_unit_sales"].fillna(0.0)

    # Attach item/store metadata.
    ev = ev.merge(items[["item_nbr", "family", "class", "perishable"]],
                  on="item_nbr", how="left")
    ev = ev.merge(stores[["store_nbr", "city", "state", "type", "cluster"]]
                  .rename(columns={"type": "store_type"}),
                  on="store_nbr", how="left")

    # Deterministic cap by descending pre-period-only volume (outcome-independent),
    # tie-broken by keys for reproducibility.
    ev = ev.sort_values(
        ["pre_period_unit_sales", "store_nbr", "item_nbr", "promo_start"],
        ascending=[False, True, True, True],
    ).head(sel.max_events)
    ev = ev.sort_values(["store_nbr", "item_nbr", "promo_start"]).reset_index(drop=True)

    # Event id and window-length columns.
    ev.insert(0, "event_id",
              [f"EVT{str(i + 1).zfill(5)}" for i in range(len(ev))])
    ev["pre_days"] = cfg.windows.pre_days
    ev["post_days"] = cfg.windows.post_days
    stats.final = len(ev)

    cols = ["event_id", "store_nbr", "item_nbr", "family", "class", "perishable",
            "promo_start", "promo_end", "promo_len", "n_promo_days",
            "pre_start", "post_end", "pre_days", "post_days",
            "nonzero_frac", "pre_period_unit_sales",
            "city", "state", "store_type", "cluster"]
    return ev[cols], stats


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def build_selected_events(
    cfg: Config, limit: int | None = None, profile: bool = True
) -> pd.DataFrame:
    """Run the full detection + selection pipeline and write the outputs.

    Args:
        cfg: loaded configuration.
        limit: optional override of ``selection.max_events`` (CLI ``--limit``).
        profile: when True, also run the one-pass ``train.csv`` profile and record the
            headline quality counts in ``selection_stats.json``.

    Returns:
        The selected-events DataFrame (also written to parquet + csv). A companion
        ``selection_stats.json`` captures the selection funnel and profiling numbers.
    """
    if limit is not None:
        cfg = _with_max_events(cfg, limit)

    out_dir = cfg.paths.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)

    con = connect_from_config(cfg)
    profile_result: dict[str, object] = {}
    try:
        if profile:
            profile_result = profile_train(cfg, con)
        series = load_candidate_series(cfg, con)
        promoted = load_promoted_rows(cfg, con)

        events = detect_events(promoted, cfg.selection.promo_flag_gap_tolerance)

        items = pd.read_csv(cfg.raw_file("items"))
        stores = pd.read_csv(cfg.raw_file("stores"))
        store_span = _store_operating_span(cfg)

        # Selection caps by pre-period-only volume, which needs the open connection.
        selected, stats = apply_selection(
            events, series, items, stores, store_span, cfg, con
        )
    finally:
        con.close()
        # Remove the project-local spill dir (created on demand by connect_from_config).
        shutil.rmtree(cfg.duckdb.temp_dir, ignore_errors=True)

    # Derived quality metrics (no extra scan): flickering events (a merged gap means the
    # flag flickered off within the run) and short-history series.
    flickering_events = int((events["n_promo_days"] < events["promo_len"]).sum())
    short_history_series = int((series["n_days"] < cfg.windows.pre_days).sum())

    logger.info("Selection funnel: raw=%d length=%d history=%d nonzero=%d clean=%d "
                "store_open=%d final=%d",
                stats.raw_events, stats.after_length, stats.after_history,
                stats.after_nonzero, stats.after_clean_window, stats.after_store_open,
                stats.final)

    parquet_path = out_dir / "selected_events.parquet"
    csv_path = out_dir / "selected_events.csv"
    selected.to_parquet(parquet_path, index=False)
    selected.to_csv(csv_path, index=False)
    logger.info("Wrote %d selected events to %s (and .csv)", len(selected), parquet_path)

    stats_payload = {
        "config": {
            "reliable_flag_start": cfg.reliable_flag_start,
            "pre_days": cfg.windows.pre_days,
            "post_days": cfg.windows.post_days,
            "min_promo_days": cfg.selection.min_promo_days,
            "max_promo_days": cfg.selection.max_promo_days,
            "min_pre_nonzero_frac": cfg.selection.min_pre_nonzero_frac,
            "promo_flag_gap_tolerance": cfg.selection.promo_flag_gap_tolerance,
            "max_events": cfg.selection.max_events,
        },
        "funnel": asdict(stats),
        "n_candidate_series": int(len(series)),
        "n_promoted_rows": int(len(promoted)),
        "flickering_events": flickering_events,
        "short_history_series": short_history_series,
        "profile": _jsonable(profile_result),
    }
    stats_path = out_dir / "selection_stats.json"
    stats_path.write_text(json.dumps(stats_payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote selection + profile stats to %s", stats_path)
    return selected


def _jsonable(d: dict[str, object]) -> dict[str, object]:
    """Convert numpy/pandas scalars in a dict to plain JSON-serializable types."""
    out: dict[str, object] = {}
    for key, value in d.items():
        if hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = str(value) if not isinstance(value, (int, float, str)) else value
    return out


def _with_max_events(cfg: Config, max_events: int) -> Config:
    """Return a copy of ``cfg`` with ``selection.max_events`` overridden."""
    from dataclasses import replace
    new_sel = replace(cfg.selection, max_events=max_events)
    return replace(cfg, selection=new_sel)


def main() -> None:
    """CLI entry point for the ``select`` / ``data`` Makefile target."""
    parser = argparse.ArgumentParser(description="Detect and select promotion events.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.yaml (defaults to config/config.yaml).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override selection.max_events (cap on analyzable events).")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    build_selected_events(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
