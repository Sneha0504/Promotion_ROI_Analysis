"""Unit 2, step 1: per-event analysis panels with donor pools.

For each selected promotion event this module builds a tidy daily panel over the event's
full ``[pre_start, post_end]`` window containing the promoted (target) item-store's sales
plus the sales of a pool of comparable, clean **donor** item-stores for synthetic control.

Pipeline
--------
1. One DuckDB streaming pass over ``train.csv`` (``sql/donor_scan.sql``) pulls every
   candidate donor row — same product *family* as each target, from the target's store and
   the stores in its cluster — bounded to the global analysis window, into a git-ignored
   parquet in ``data/interim``.
2. Each event is processed in pandas:
   * both target and donor series are reindexed to the **full daily calendar** (absent days
     are genuine zero demand, because Favorita stores no zero-sales rows) and then
     **store-closure days are masked** (a store-day with no transactions row is a closure,
     set to NaN rather than zero — see :mod:`src.features`);
   * negative ``unit_sales`` (returns) are clipped to 0 for demand modeling (documented
     data-quality rule 1); the raw negative count is logged;
   * donors are filtered to be **clean controls** — never on promotion anywhere in the
     window — with complete, non-degenerate pre-period sales, then ranked by pre-period
     correlation with the target and capped at ``donors.max_donors``;
   * the pool is drawn from the **same store first** and expanded to **same-cluster**
     stores only when it is short of ``donors.min_donors``.
3. Events that cannot reach ``donors.min_donors`` valid donors are **dropped** and the
   reason recorded; the analyzable count can therefore fall below the 300 selected events.

Outputs
-------
* ``data/interim/donor_scan.parquet`` — raw candidate donor rows (git-ignored, large).
* ``data/interim/analysis_panels.parquet`` — long per-event target+donor daily series
  (git-ignored input to the model units).
* ``data/processed/analyzable_events.parquet`` — one row per surviving event with donor
  metadata (committed).
* ``data/processed/dropped_events.parquet`` — dropped events with reasons (committed).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.features import closure_mask, full_calendar, load_open_days
from src.load import connect_from_config, read_sql
from src.logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Candidate-pair construction + donor scan
# --------------------------------------------------------------------------------------
def build_family_items(items: pd.DataFrame, cfg: Config) -> dict[object, np.ndarray]:
    """Map each donor similarity key to its array of ``item_nbr``.

    The key is the product ``family``, or the ``(family, class)`` tuple when
    ``donors.same_class`` is enabled.
    """
    if cfg.donors.same_class:
        return {k: g["item_nbr"].to_numpy()
                for k, g in items.groupby(["family", "class"])}
    return {k: g["item_nbr"].to_numpy() for k, g in items.groupby("family")}


def _event_family_key(event: pd.Series, cfg: Config) -> object:
    """Donor similarity key for a single event."""
    if cfg.donors.same_class:
        return (event["family"], int(event["class"]))
    return event["family"]


def build_candidate_pairs(
    events: pd.DataFrame, fam_items: dict[object, np.ndarray],
    stores: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Return the unique (store_nbr, item_nbr) donor candidate pairs across all events.

    Candidates share each target's product ``family`` (and ``class`` when
    ``donors.same_class`` is set) and are drawn from the target's own store plus, when
    cluster expansion is enabled, the stores in the target's cluster.
    """
    don = cfg.donors
    cluster_stores = stores.groupby("cluster")["store_nbr"].apply(list).to_dict()
    store_cluster = stores.set_index("store_nbr")["cluster"].to_dict()

    pairs: set[tuple[int, int]] = set()
    for _, event in events.iterrows():
        key = _event_family_key(event, cfg)
        if key not in fam_items:
            continue
        store_set = {int(event["store_nbr"])}
        if don.expand_to_cluster:
            store_set.update(
                int(s) for s in cluster_stores[store_cluster[event["store_nbr"]]]
            )
        for store in store_set:
            for item in fam_items[key]:
                pairs.add((store, int(item)))

    pairs_df = pd.DataFrame(sorted(pairs), columns=["store_nbr", "item_nbr"])
    logger.info("Donor candidate pairs: %s (across %d events)",
                f"{len(pairs_df):,}", len(events))
    return pairs_df


def run_donor_scan(
    cfg: Config, pairs: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> Path:
    """Stream the candidate donor rows to ``data/interim/donor_scan.parquet``.

    One DuckDB pass over ``train.csv`` inner-joined to the registered ``donor_pairs``
    relation; the result is written with ``COPY ... TO`` so nothing large is held in RAM.
    Returns the output path.
    """
    out_path = cfg.paths.data_interim / "donor_scan.parquet"
    cfg.paths.data_interim.mkdir(parents=True, exist_ok=True)

    con = connect_from_config(cfg)
    con.register("donor_pairs", pairs)
    try:
        select_sql = read_sql(cfg, "donor_scan")
        copy_sql = (
            f"COPY ({select_sql.rstrip().rstrip(';')}) "
            f"TO '{out_path.as_posix()}' (FORMAT PARQUET)"
        )
        logger.info("Scanning train.csv for donor rows -> %s", out_path.name)
        con.execute(copy_sql, [str(cfg.train_path()), str(start.date()), str(end.date())])
    finally:
        con.unregister("donor_pairs")
        con.close()

    rows = pd.read_parquet(out_path, columns=["store_nbr"]).shape[0]
    size_mb = out_path.stat().st_size / 1e6
    logger.info("Donor scan complete: %s rows, %.0f MB", f"{rows:,}", size_mb)
    return out_path


# --------------------------------------------------------------------------------------
# Dense-calendar + closure masking
# --------------------------------------------------------------------------------------
def dense_series(
    raw: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    open_days: dict[int, set[pd.Timestamp]],
    store_nbr: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build dense (calendar x item) sales and on-promotion frames for one store.

    ``raw`` holds the scanned rows for a single store over the window. The sales frame is
    reindexed to ``calendar``; absent cells on **open** days become 0 demand while cells on
    **closure** days (store had no transactions) are left as NaN (masked). Negative sales
    are clipped to 0. The on-promotion frame marks any day/item flagged on promotion.

    Returns ``(sales, onpromo)`` with dates as the index and ``item_nbr`` as columns.
    """
    sales = raw.pivot_table(index="date", columns="item_nbr", values="unit_sales",
                            aggfunc="sum")
    promo = raw.pivot_table(index="date", columns="item_nbr", values="onpromotion",
                            aggfunc="max")

    sales = sales.reindex(calendar)
    # ``eq(True)`` maps both NaN (absent day) and False to False, giving a clean bool frame.
    promo = promo.reindex(calendar).eq(True)

    closed = closure_mask(open_days, store_nbr, calendar)
    open_idx = calendar[~closed]
    closed_idx = calendar[closed]

    # Open days: absent (store, item, date) is genuine zero demand.
    sales.loc[open_idx] = sales.loc[open_idx].fillna(0.0)
    # Closure days: mask every item (no genuine demand observable).
    if len(closed_idx) > 0:
        sales.loc[closed_idx] = np.nan

    sales = sales.clip(lower=0.0)  # returns (negatives) -> 0; NaN closures preserved
    return sales, promo


# --------------------------------------------------------------------------------------
# Donor selection for one event
# --------------------------------------------------------------------------------------
@dataclass
class DonorSelection:
    """Result of selecting donors for one event."""

    donor_items: list[tuple[int, int]]           # (store_nbr, item_nbr)
    donor_corr: dict[tuple[int, int], float]      # pre-period correlation to target
    source: str                                   # 'same_store' | 'cluster_expanded'
    n_candidates: int
    reason: str | None                            # non-None => dropped


def _valid_donors(
    sales: pd.DataFrame,
    promo: pd.DataFrame,
    target: pd.Series,
    pre_mask: np.ndarray,
    cfg: Config,
) -> dict[int, float]:
    """Return {item_nbr: pre-period correlation} for donors passing every clean-control test.

    A donor is kept when it is never on promotion in the window, sells on at least
    ``donors.min_donor_pre_nonzero_frac`` of its observed pre-period days, has non-degenerate
    pre-period variation, and correlates with the target at >= ``donors.min_donor_pre_corr``.
    """
    don = cfg.donors
    target_pre = target.to_numpy()[pre_mask]
    target_valid = ~np.isnan(target_pre)
    out: dict[int, float] = {}

    for item in sales.columns:
        # Clean control: never flagged on promotion anywhere in the window.
        if bool(promo[item].to_numpy().any()):
            continue
        col = sales[item].to_numpy()
        col_pre = col[pre_mask]
        both = target_valid & ~np.isnan(col_pre)
        if both.sum() < 2:
            continue
        donor_vals = col_pre[both]
        # Non-degenerate and sufficiently active in the pre-period.
        nonzero_frac = float((donor_vals > 0).mean())
        if nonzero_frac < don.min_donor_pre_nonzero_frac:
            continue
        if np.std(donor_vals) <= 0:
            continue
        t_vals = target_pre[both]
        if np.std(t_vals) <= 0:
            continue
        corr = float(np.corrcoef(t_vals, donor_vals)[0, 1])
        if np.isnan(corr) or corr < don.min_donor_pre_corr:
            continue
        out[int(item)] = corr
    return out


def select_donors_for_event(
    event: pd.Series,
    scan_by_store: dict[int, pd.DataFrame],
    open_days: dict[int, set[pd.Timestamp]],
    stores: pd.DataFrame,
    fam_items: dict[object, np.ndarray],
    cfg: Config,
) -> tuple[DonorSelection, pd.DataFrame | None]:
    """Select donors for one event and return (selection, panel_long_or_None).

    The panel is ``None`` when the event is dropped. Donors are taken from the same store
    first and only expanded to same-cluster stores when the same-store pool falls short of
    ``donors.min_donors``.
    """
    don = cfg.donors
    calendar = full_calendar(event.pre_start, event.post_end)
    pre_mask = (calendar >= event.pre_start) & (calendar < event.promo_start)
    promo_mask = (calendar >= event.promo_start) & (calendar <= event.promo_end)
    post_mask = calendar > event.promo_end

    # Candidate family items (exclude the target item itself).
    key = _event_family_key(event, cfg)
    all_fam_items = set(int(i) for i in np.atleast_1d(fam_items[key]))
    all_fam_items.discard(int(event.item_nbr))

    # --- target dense series (from the target's own store slice) ------------------------
    store_raw = scan_by_store.get(int(event.store_nbr))
    if store_raw is None:
        return (DonorSelection([], {}, "none", 0, "no_store_rows"), None)
    tgt_raw = store_raw[store_raw["item_nbr"] == int(event.item_nbr)]
    tgt_sales, _ = dense_series(tgt_raw, calendar, open_days, int(event.store_nbr))
    if int(event.item_nbr) not in tgt_sales.columns:
        return (DonorSelection([], {}, "none", 0, "no_target_rows"), None)
    target = tgt_sales[int(event.item_nbr)]
    # Target must be modelable: enough open (non-masked) pre-period days with variation.
    tgt_pre = target.to_numpy()[pre_mask]
    tgt_pre_valid = tgt_pre[~np.isnan(tgt_pre)]
    if tgt_pre_valid.size < 2 or np.std(tgt_pre_valid) <= 0:
        return (DonorSelection([], {}, "none", 0, "degenerate_target"), None)

    store_cluster = stores.set_index("store_nbr")["cluster"].to_dict()
    cluster_stores = stores.groupby("cluster")["store_nbr"].apply(list).to_dict()

    def gather(store_list: list[int]) -> dict[tuple[int, int], float]:
        found: dict[tuple[int, int], float] = {}
        for store in store_list:
            raw = scan_by_store.get(int(store))
            if raw is None:
                continue
            raw = raw[raw["item_nbr"].isin(all_fam_items)]
            if raw.empty:
                continue
            d_sales, d_promo = dense_series(raw, calendar, open_days, int(store))
            corrs = _valid_donors(d_sales, d_promo, target, pre_mask, cfg)
            for item, corr in corrs.items():
                found[(int(store), int(item))] = corr
        return found

    # Same store first.
    donors = gather([int(event.store_nbr)])
    source = "same_store"
    n_candidates = len(donors)

    # Expand to cluster stores only if short of the minimum.
    if len(donors) < don.min_donors and don.expand_to_cluster:
        other_stores = [int(s) for s in cluster_stores[store_cluster[event.store_nbr]]
                        if int(s) != int(event.store_nbr)]
        extra = gather(other_stores)
        if extra:
            donors.update(extra)
            source = "cluster_expanded"
            n_candidates = len(donors)

    if len(donors) < don.min_donors:
        return (DonorSelection([], {}, source, n_candidates,
                               f"insufficient_donors({len(donors)})"), None)

    # Rank by descending pre-period correlation and cap the pool.
    ranked = sorted(donors.items(), key=lambda kv: kv[1], reverse=True)[: don.max_donors]
    donor_items = [k for k, _ in ranked]
    donor_corr = {k: v for k, v in ranked}

    panel = _emit_panel(event, calendar, pre_mask, promo_mask, post_mask,
                        target, donor_items, donor_corr, scan_by_store, open_days)
    return (DonorSelection(donor_items, donor_corr, source, n_candidates, None), panel)


def _segment(pre_mask: np.ndarray, promo_mask: np.ndarray,
             post_mask: np.ndarray) -> np.ndarray:
    """Per-day segment label array ('pre'/'promo'/'post')."""
    seg = np.empty(len(pre_mask), dtype=object)
    seg[pre_mask] = "pre"
    seg[promo_mask] = "promo"
    seg[post_mask] = "post"
    return seg


def _emit_panel(
    event: pd.Series,
    calendar: pd.DatetimeIndex,
    pre_mask: np.ndarray,
    promo_mask: np.ndarray,
    post_mask: np.ndarray,
    target: pd.Series,
    donor_items: list[tuple[int, int]],
    donor_corr: dict[tuple[int, int], float],
    scan_by_store: dict[int, pd.DataFrame],
    open_days: dict[int, set[pd.Timestamp]],
) -> pd.DataFrame:
    """Build the long per-event panel (target + ranked donors) for one event."""
    seg = _segment(pre_mask, promo_mask, post_mask)
    frames: list[pd.DataFrame] = []

    tgt_vals = target.to_numpy()
    frames.append(pd.DataFrame({
        "event_id": event.event_id,
        "date": calendar,
        "store_nbr": int(event.store_nbr),
        "item_nbr": int(event.item_nbr),
        "role": "target",
        "donor_rank": 0,
        "pre_corr": np.nan,
        "unit_sales": tgt_vals,
        "is_closure": np.isnan(tgt_vals),
        "segment": seg,
    }))

    # Recompute donor dense series by store (small: <= max_donors columns).
    by_store: dict[int, list[int]] = {}
    for store, item in donor_items:
        by_store.setdefault(store, []).append(item)

    dense_cache: dict[int, pd.DataFrame] = {}
    for store, items in by_store.items():
        raw = scan_by_store[store]
        raw = raw[raw["item_nbr"].isin(items)]
        d_sales, _ = dense_series(raw, calendar, open_days, store)
        dense_cache[store] = d_sales

    for rank, (store, item) in enumerate(donor_items, start=1):
        vals = dense_cache[store][item].to_numpy()
        frames.append(pd.DataFrame({
            "event_id": event.event_id,
            "date": calendar,
            "store_nbr": store,
            "item_nbr": item,
            "role": "donor",
            "donor_rank": rank,
            "pre_corr": donor_corr[(store, item)],
            "unit_sales": vals,
            "is_closure": np.isnan(vals),
            "segment": seg,
        }))

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def build_panels(cfg: Config, limit: int | None = None,
                 reuse_scan: bool = False) -> pd.DataFrame:
    """Build donor panels for every selected event; write panels + survivor/drop tables.

    Args:
        cfg: loaded configuration.
        limit: process only the first ``limit`` events (debugging aid).
        reuse_scan: reuse an existing ``donor_scan.parquet`` instead of re-scanning.

    Returns:
        The analyzable-events DataFrame (survivors with donor metadata).
    """
    events = pd.read_parquet(cfg.paths.data_processed / "selected_events.parquet")
    if limit is not None:
        events = events.head(limit).copy()
    items = pd.read_csv(cfg.raw_file("items"))
    stores = pd.read_csv(cfg.raw_file("stores"))

    gmin = events["pre_start"].min()
    gmax = events["post_end"].max()
    fam_items = build_family_items(items, cfg)

    scan_path = cfg.paths.data_interim / "donor_scan.parquet"
    if not (reuse_scan and scan_path.exists()):
        pairs = build_candidate_pairs(events, fam_items, stores, cfg)
        run_donor_scan(cfg, pairs, gmin, gmax)
    else:
        logger.info("Reusing existing donor scan at %s", scan_path)

    logger.info("Loading donor scan into memory and indexing by store...")
    scan = pd.read_parquet(scan_path)
    scan["date"] = pd.to_datetime(scan["date"])
    n_negative = int((scan["unit_sales"] < 0).sum())
    logger.info("Donor scan: %s rows; negative (return) rows clipped to 0: %s",
                f"{len(scan):,}", f"{n_negative:,}")
    scan_by_store = {int(s): g for s, g in scan.groupby("store_nbr", sort=False)}
    del scan

    open_days = load_open_days(cfg)

    panels: list[pd.DataFrame] = []
    survivors: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []

    for i, (_, event) in enumerate(events.iterrows(), start=1):
        selection, panel = select_donors_for_event(
            event, scan_by_store, open_days, stores, fam_items, cfg
        )
        if panel is None:
            dropped.append({"event_id": event.event_id, "family": event.family,
                            "store_nbr": int(event.store_nbr),
                            "item_nbr": int(event.item_nbr),
                            "n_candidates": selection.n_candidates,
                            "reason": selection.reason})
        else:
            panels.append(panel)
            corrs = list(selection.donor_corr.values())
            survivors.append({
                **{c: event[c] for c in events.columns},
                "n_donors": len(selection.donor_items),
                "donor_source": selection.source,
                "n_donor_candidates": selection.n_candidates,
                "mean_donor_pre_corr": float(np.mean(corrs)) if corrs else np.nan,
                "max_donor_pre_corr": float(np.max(corrs)) if corrs else np.nan,
            })
        if i % 25 == 0 or i == len(events):
            logger.info("Processed %d/%d events (kept=%d dropped=%d)",
                        i, len(events), len(survivors), len(dropped))

    out_proc = cfg.paths.data_processed
    out_proc.mkdir(parents=True, exist_ok=True)
    cfg.paths.data_interim.mkdir(parents=True, exist_ok=True)

    analyzable = pd.DataFrame(survivors)
    dropped_df = pd.DataFrame(
        dropped, columns=["event_id", "family", "store_nbr", "item_nbr",
                          "n_candidates", "reason"]
    )
    analyzable.to_parquet(out_proc / "analyzable_events.parquet", index=False)
    dropped_df.to_parquet(out_proc / "dropped_events.parquet", index=False)

    if panels:
        panel_all = pd.concat(panels, ignore_index=True)
        panel_all.to_parquet(cfg.paths.data_interim / "analysis_panels.parquet",
                             index=False)
        logger.info("Wrote %s panel rows to data/interim/analysis_panels.parquet",
                    f"{len(panel_all):,}")

    logger.info("Donor-pool build complete: %d analyzable events, %d dropped (of %d)",
                len(analyzable), len(dropped_df), len(events))
    if not dropped_df.empty:
        reason_counts = dropped_df["reason"].str.replace(r"\(\d+\)", "", regex=True)
        logger.info("Drop reasons: %s", reason_counts.value_counts().to_dict())
    return analyzable


def main() -> None:
    """CLI entry point for the ``panel`` Makefile target."""
    parser = argparse.ArgumentParser(description="Build per-event donor panels.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N selected events.")
    parser.add_argument("--reuse-scan", action="store_true",
                        help="Reuse an existing donor_scan.parquet instead of re-scanning.")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    build_panels(cfg, limit=args.limit, reuse_scan=args.reuse_scan)


if __name__ == "__main__":
    main()
