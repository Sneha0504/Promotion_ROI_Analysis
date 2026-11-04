"""Unit 3, step 1: cannibalization (PLAN sec. 6c).

For every analyzable event this module picks a single **preferred effect estimate** and
quantifies cannibalization -- the post-promotion sales dip expressed as a share of the
promotion-window lift.

Method selection (documented, recorded per event)
-------------------------------------------------
* Prefer **synthetic control** when its pre-period fit is good (``synth poor_fit == False``);
  the two independent counterfactuals agree strongly on the promotion-window lift (Unit 2),
  and SC matches levels directly without a log back-transform.
* Otherwise **fall back to interrupted time series** (ITS).
* When **both** methods are flagged poor-fit the event is still computed but marked
  ``low_confidence`` so downstream consumers can treat it with caution.

Cannibalization definition (documented)
---------------------------------------
``cannibalization_units = max(0, -post_effect)`` -- the magnitude of the post-period dip in
original units (a positive post effect means no pull-forward, so 0 units).

``cannibalization_share = max(0, -post_effect) / promo_effect`` when ``promo_effect > 0``:
0% = no pull-forward, ~100% = the entire promo gain was borrowed from the future. The share
is intentionally left un-capped (a value above 1 means the post dip exceeded the lift).

Edge cases (explicit):
* ``promo_effect <= 0`` -- there was no positive lift to borrow against, so a cannibalization
  *share* is undefined; it is recorded as NaN (the ``cannibalization_units`` dip is still
  reported). These events cannot be a naive "win" anyway.
* ``post_effect > 0`` (sales rose post-promotion rather than dipping) -- no cannibalization:
  ``cannibalization_units = 0`` and ``cannibalization_share = 0``.

A ``post_near_zero`` flag marks events whose total post-window actual sales are at or below
``cannibalization.post_near_zero_units``: a near-zero post window is more likely a delisting
or stockout than genuine pull-forward, so its cannibalization figure must not be
over-interpreted.

All effect inputs are the Unit-2 **percentile** bootstrap point estimates; the CIs they carry
are percentile intervals and are labeled as such wherever surfaced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.logging_utils import get_logger

logger = get_logger(__name__)


def choose_method(sc_poor_fit: bool, its_poor_fit: bool) -> tuple[str, bool]:
    """Pick the preferred estimator for one event and flag low-confidence cases.

    Args:
        sc_poor_fit: whether synthetic control is flagged poor pre-fit.
        its_poor_fit: whether interrupted time series is flagged poor pre-fit.

    Returns:
        ``(chosen_method, low_confidence)`` where ``chosen_method`` is ``"SC"`` when the
        synthetic-control pre-fit is good, else ``"ITS"``; ``low_confidence`` is True only
        when *both* methods are flagged poor-fit.
    """
    chosen = "SC" if not sc_poor_fit else "ITS"
    low_confidence = bool(sc_poor_fit and its_poor_fit)
    return chosen, low_confidence


def cannibalization_share(promo_effect: float, post_effect: float) -> float:
    """Return the cannibalization share for one event (see module docstring).

    ``max(0, -post_effect) / promo_effect`` when ``promo_effect > 0``; NaN when the lift is
    non-positive (a share of a non-existent lift is undefined). A positive ``post_effect``
    yields 0 (no pull-forward).
    """
    if not np.isfinite(promo_effect) or promo_effect <= 0:
        return float("nan")
    dip = max(0.0, -post_effect)
    return dip / promo_effect


def build_cannibalization(its: pd.DataFrame, synth: pd.DataFrame,
                          post_near_zero_units: float) -> pd.DataFrame:
    """Build the per-event cannibalization table from the ITS and SC result tables.

    ``post_near_zero_units`` flags events whose total post-window actual sales are at or below
    the threshold: such a "dip" is more likely a delisting / stockout than genuine
    pull-forward, so its cannibalization figure should not be over-interpreted.
    """
    its_cols = its[["event_id", "poor_fit", "promo_effect", "post_effect",
                    "post_actual"]].rename(
        columns={"poor_fit": "its_poor_fit", "promo_effect": "its_promo_effect",
                 "post_effect": "its_post_effect", "post_actual": "its_post_actual"})
    sc_cols = synth[["event_id", "poor_fit", "promo_effect", "post_effect",
                     "post_actual"]].rename(
        columns={"poor_fit": "sc_poor_fit", "promo_effect": "sc_promo_effect",
                 "post_effect": "sc_post_effect", "post_actual": "sc_post_actual"})
    merged = its_cols.merge(sc_cols, on="event_id", how="inner")

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        chosen, low_conf = choose_method(bool(row.sc_poor_fit), bool(row.its_poor_fit))
        if chosen == "SC":
            promo_effect = float(row.sc_promo_effect)
            post_effect = float(row.sc_post_effect)
            post_actual = float(row.sc_post_actual)
        else:
            promo_effect = float(row.its_promo_effect)
            post_effect = float(row.its_post_effect)
            post_actual = float(row.its_post_actual)
        rows.append({
            "event_id": row.event_id,
            "chosen_method": chosen,
            "promo_effect": promo_effect,
            "post_effect": post_effect,
            "post_actual": post_actual,
            "cannibalization_units": max(0.0, -post_effect),
            "cannibalization_share": cannibalization_share(promo_effect, post_effect),
            "post_near_zero": bool(post_actual <= post_near_zero_units),
            "low_confidence": low_conf,
        })
    return pd.DataFrame(rows)


def run_cannibalization(cfg: Config) -> pd.DataFrame:
    """Compute and write the per-event cannibalization table."""
    out = cfg.paths.data_processed
    its = pd.read_parquet(out / "its_results.parquet")
    synth = pd.read_parquet(out / "synth_results.parquet")

    cannib = build_cannibalization(its, synth, cfg.cannibalization.post_near_zero_units)
    out.mkdir(parents=True, exist_ok=True)
    cannib.to_parquet(out / "cannibalization.parquet", index=False)

    method_counts = cannib["chosen_method"].value_counts().to_dict()
    median_share = float(cannib["cannibalization_share"].median(skipna=True))
    logger.info("Cannibalization complete: %d events | chosen method %s | "
                "%d low-confidence (both poor-fit) | %d post-near-zero (possible "
                "delisting/stockout)", len(cannib), method_counts,
                int(cannib["low_confidence"].sum()), int(cannib["post_near_zero"].sum()))
    logger.info("Median cannibalization share (where defined): %.1f%%", 100 * median_share)
    return cannib


def main() -> None:
    """CLI entry point (part of the ``impact`` Makefile target)."""
    parser = argparse.ArgumentParser(description="Compute cannibalization per promotion.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    run_cannibalization(cfg)


if __name__ == "__main__":
    main()
