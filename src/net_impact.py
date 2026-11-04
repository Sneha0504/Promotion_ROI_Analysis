"""Unit 3, step 2: net impact + classification (PLAN sec. 6d, Step 8).

Combines the promotion-window lift and the post-period cannibalization dip into a single
**net incremental units** figure, converts it to a **net profit impact** under clearly
labeled economic assumptions, and classifies every promotion as net positive, net negative
or inconclusive.

Sign convention (documented)
----------------------------
Effects are signed ``actual - counterfactual``. A post-period dip is therefore a *negative*
``post_effect``. Net incremental units are the straight signed sum::

    net_incremental_units = promo_effect + post_effect

which equals ``promo_lift - cannibalization_units`` when the post window is a dip
(``post_effect <= 0``), the usual case.

Economic assumptions (Favorita has NO price/margin data -- these are config assumptions)
--------------------------------------------------------------------------------------
``assumed_price`` is normalized to 1.0 so results read in margin-units.

* ``discount_cost       = discount_depth * assumed_price * promo_units_sold`` where
  ``promo_units_sold`` is the **actual** units sold during the promotion window (the discount
  is paid on every unit actually sold at the promo price, not only incremental ones).
* ``incremental_margin  = gross_margin * assumed_price * net_incremental_units``
* ``net_profit_impact   = incremental_margin - discount_cost``

Classification (documented)
---------------------------
The net-effect confidence interval is propagated from the **chosen method's** promo + post
**percentile** CIs by summing their bounds::

    net_units_lo = promo_effect_lo + post_effect_lo
    net_units_hi = promo_effect_hi + post_effect_hi

This is the conservative (widest) combination -- it treats the two windows' errors as moving
together, so it is cautious about declaring a promotion conclusively positive or negative.
The net-profit CI is the affine (monotone increasing) image of the net-units CI, because the
discount cost is a fixed offset (based on actual units sold, not on the uncertain effect)::

    net_profit_lo = gross_margin * assumed_price * net_units_lo - discount_cost
    net_profit_hi = gross_margin * assumed_price * net_units_hi - discount_cost

A promotion is **inconclusive** when the chosen straddle-test interval contains zero; else it
is **net positive** / **net negative** by the sign of ``net_profit_impact``. Events flagged
``low_confidence`` (both methods poor-fit) are **forced to inconclusive** regardless of their
CI, since a both-poor-fit estimate cannot support a confident call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.logging_utils import get_logger

logger = get_logger(__name__)


def compute_economics(
    promo_effect: float,
    post_effect: float,
    promo_units_sold: float,
    *,
    assumed_price: float,
    discount_depth: float,
    gross_margin: float,
) -> dict[str, float]:
    """Return the net-impact economics for one promotion (pure, testable).

    See the module docstring for the definitions and sign convention. All price/margin inputs
    are modeled assumptions.
    """
    net_incremental_units = promo_effect + post_effect
    discount_cost = discount_depth * assumed_price * promo_units_sold
    incremental_margin = gross_margin * assumed_price * net_incremental_units
    net_profit_impact = incremental_margin - discount_cost
    return {
        "net_incremental_units": net_incremental_units,
        "discount_cost": discount_cost,
        "incremental_margin": incremental_margin,
        "net_profit_impact": net_profit_impact,
    }


def compute_break_even(
    net_incremental_units: float,
    promo_units_sold: float,
    *,
    discount_depth: float,
    gross_margin: float,
) -> dict[str, float | bool]:
    """Per-event break-even economics (pure, testable).

    * ``break_even_discount`` = ``gross_margin * net_incremental_units / promo_units_sold`` -- the
      discount depth at which net profit is exactly zero, holding margin fixed. A promotion run at
      a *shallower* discount than this is net positive.
    * ``break_even_margin`` = ``discount_depth * promo_units_sold / net_incremental_units`` -- the
      gross margin required to break even at the *current* discount depth.
    * ``never_breaks_even`` is True when ``net_incremental_units <= 0``: the promotion created no net
      incremental volume, so no discount/margin makes it profitable (break_even_margin is undefined
      and break_even_discount is non-positive).
    """
    never = net_incremental_units <= 0
    be_discount = (gross_margin * net_incremental_units / promo_units_sold
                   if promo_units_sold > 0 else float("nan"))
    be_margin = (discount_depth * promo_units_sold / net_incremental_units
                 if net_incremental_units > 0 else float("nan"))
    return {
        "break_even_discount": be_discount,
        "break_even_margin": be_margin,
        "never_breaks_even": never,
    }


def net_profit_after_shift(
    net_incremental_units: float, placebo_shift: float, discount_cost: float, gm_price: float,
) -> float:
    """Net profit if the net incremental units are shifted by ``placebo_shift`` (pure, testable).

    Used by the placebo-calibrated robustness check: adding an optimistic (positive) placebo net
    effect to the measured net incremental units and recomputing net profit. ``gm_price`` is
    ``gross_margin * assumed_price``; the discount cost is a fixed offset.
    """
    return gm_price * (net_incremental_units + placebo_shift) - discount_cost


def classify(lo: float, hi: float, point: float) -> str:
    """Classify a promotion from its net-effect CI and point estimate.

    ``inconclusive`` when the interval ``[lo, hi]`` straddles zero; otherwise ``net positive``
    or ``net negative`` by the sign of ``point``.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if lo <= 0.0 <= hi:
        return "inconclusive"
    return "net positive" if point > 0 else "net negative"


def _chosen_bounds(cannib: pd.DataFrame, its: pd.DataFrame,
                   synth: pd.DataFrame) -> pd.DataFrame:
    """Attach the chosen method's promo/post percentile CI bounds to each event."""
    its_b = its[["event_id", "promo_effect_lo", "promo_effect_hi",
                 "post_effect_lo", "post_effect_hi"]].add_prefix("its_")
    sc_b = synth[["event_id", "promo_effect_lo", "promo_effect_hi",
                  "post_effect_lo", "post_effect_hi"]].add_prefix("sc_")
    merged = cannib.merge(its_b, left_on="event_id", right_on="its_event_id", how="left")
    merged = merged.merge(sc_b, left_on="event_id", right_on="sc_event_id", how="left")

    is_sc = merged["chosen_method"] == "SC"
    for field in ("promo_effect_lo", "promo_effect_hi", "post_effect_lo", "post_effect_hi"):
        merged[field] = np.where(is_sc, merged[f"sc_{field}"], merged[f"its_{field}"])
    return merged.drop(columns=["its_event_id", "sc_event_id"])


def _promo_units_sold(display_series: pd.DataFrame) -> pd.Series:
    """Actual units sold during each event's promotion window (nan-safe, from the panel)."""
    promo = display_series[display_series["in_promo"]]
    return promo.groupby("event_id")["actual"].sum(min_count=1)


def _classify_frame(frame: pd.DataFrame, basis: str) -> pd.Series:
    """Classify every row of ``frame`` using the configured straddle-test basis.

    Events flagged ``low_confidence`` (both methods poor-fit, including an ITS-implausible
    counterfactual when ITS is the chosen estimate) are forced to ``inconclusive`` -- a
    both-poor-fit estimate cannot support a confident net-positive/negative call (PLAN sec. 6d:
    "inconclusive = wide / CI straddles zero").
    """
    if basis == "net_units":
        lo, hi, point = (frame["net_units_lo"], frame["net_units_hi"],
                         frame["net_incremental_units"])
    else:  # "net_profit"
        lo, hi, point = (frame["net_profit_lo"], frame["net_profit_hi"],
                         frame["net_profit_impact"])
    labels = pd.Series(
        [classify(lo_i, hi_i, p_i) for lo_i, hi_i, p_i in zip(lo, hi, point, strict=True)],
        index=frame.index,
    )
    if "low_confidence" in frame.columns:
        labels = labels.mask(frame["low_confidence"].astype(bool), "inconclusive")
    return labels


def _split_counts(labels: pd.Series) -> dict[str, object]:
    """Return counts and percentages of the three classes."""
    order = ["net positive", "net negative", "inconclusive"]
    counts = {c: int((labels == c).sum()) for c in order}
    total = int(len(labels))
    pct = {c: round(100.0 * counts[c] / total, 1) if total else 0.0 for c in order}
    return {"counts": counts, "pct": pct, "total": total}


def build_net_impact(cfg: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute per-event net impact and the summary (split + sensitivity grid)."""
    out = cfg.paths.data_processed
    cannib = pd.read_parquet(out / "cannibalization.parquet")
    its = pd.read_parquet(out / "its_results.parquet")
    synth = pd.read_parquet(out / "synth_results.parquet")
    display_series = pd.read_parquet(out / "display_series.parquet")

    frame = _chosen_bounds(cannib, its, synth)
    units_sold = _promo_units_sold(display_series)
    frame["promo_units_sold"] = frame["event_id"].map(units_sold).astype(float)

    econ = cfg.economics
    base = frame.apply(
        lambda r: compute_economics(
            r["promo_effect"], r["post_effect"], r["promo_units_sold"],
            assumed_price=econ.assumed_price, discount_depth=econ.discount_depth,
            gross_margin=econ.gross_margin),
        axis=1, result_type="expand")
    frame = pd.concat([frame, base], axis=1)

    # Net-effect CI, propagated from the chosen method's promo + post percentile bounds.
    frame["net_units_lo"] = frame["promo_effect_lo"] + frame["post_effect_lo"]
    frame["net_units_hi"] = frame["promo_effect_hi"] + frame["post_effect_hi"]
    gm_price = econ.gross_margin * econ.assumed_price
    frame["net_profit_lo"] = gm_price * frame["net_units_lo"] - frame["discount_cost"]
    frame["net_profit_hi"] = gm_price * frame["net_units_hi"] - frame["discount_cost"]

    frame["classification"] = _classify_frame(frame, cfg.classification.inconclusive_basis)

    # Break-even economics per event (how far the verdict depends on the 20%/30% assumptions).
    be = frame.apply(
        lambda r: compute_break_even(
            r["net_incremental_units"], r["promo_units_sold"],
            discount_depth=econ.discount_depth, gross_margin=econ.gross_margin),
        axis=1, result_type="expand")
    frame = pd.concat([frame, be], axis=1)

    # SC placebo credibility + ITS fit quality (additive; do not change primary labels).
    syn_i = synth.set_index("event_id")
    frame["sc_low_credibility"] = frame["event_id"].map(
        syn_i["sc_low_credibility"]).fillna(False).astype(bool)
    frame["placebo_net_p90"] = frame["event_id"].map(syn_i["placebo_net_p90"]).astype(float)
    frame["placebo_net_max"] = frame["event_id"].map(syn_i["placebo_net_max"]).astype(float)
    its_poor = its.set_index("event_id")["poor_fit"]
    frame["its_poor_fit"] = frame["event_id"].map(its_poor).fillna(True).astype(bool)
    # An estimate is "credible" when the chosen counterfactual is itself trustworthy: a
    # chosen SC that is not placebo-low-credibility, or a chosen ITS with a good pre-fit.
    is_sc = frame["chosen_method"] == "SC"
    frame["estimate_credible"] = np.where(
        is_sc, ~frame["sc_low_credibility"], ~frame["its_poor_fit"]).astype(bool)

    columns = [
        "event_id", "chosen_method", "low_confidence", "post_near_zero", "sc_low_credibility",
        "estimate_credible",
        "promo_effect", "post_effect", "cannibalization_units", "cannibalization_share",
        "promo_units_sold", "net_incremental_units", "discount_cost", "incremental_margin",
        "net_profit_impact", "net_profit_lo", "net_profit_hi",
        "break_even_discount", "break_even_margin", "never_breaks_even",
        "placebo_net_p90", "placebo_net_max", "classification",
    ]
    # Rename for the output schema: promo_effect is the promotion-window lift.
    net_impact = frame[columns].rename(columns={
        "promo_effect": "lift", "cannibalization_units": "cannibalization"})

    # Overall split + how many used SC vs ITS.
    summary: dict[str, object] = {
        "assumptions": {
            "assumed_price": econ.assumed_price,
            "discount_depth": econ.discount_depth,
            "gross_margin": econ.gross_margin,
            "note": ("Favorita has no price/margin data; price/discount/margin are MODELED "
                     "assumptions. Results read in margin-units at assumed_price=1.0."),
        },
        "ci_note": ("Effect CIs are Unit-2 PERCENTILE bootstrap intervals; the net-effect CI "
                    "sums the chosen method's promo + post bounds (conservative/widest)."),
        "inconclusive_basis": cfg.classification.inconclusive_basis,
        "overall_split": _split_counts(net_impact["classification"]),
        "overall_split_excl_low_confidence": _split_counts(
            net_impact.loc[~net_impact["low_confidence"], "classification"]),
        "chosen_method_counts": {
            k: int(v) for k, v in net_impact["chosen_method"].value_counts().items()},
        "median_net_profit_impact": float(net_impact["net_profit_impact"].median()),
        "mean_net_profit_impact": float(net_impact["net_profit_impact"].mean()),
        "mean_net_profit_impact_excl_low_confidence": float(
            net_impact.loc[~net_impact["low_confidence"], "net_profit_impact"].mean()),
        "n_low_confidence": int(net_impact["low_confidence"].sum()),
        "n_post_near_zero": int(net_impact["post_near_zero"].sum()),
    }

    # "Naive win but net negative" (Key Question 3): positive promo-window lift, yet classified
    # net negative once cannibalization + discount are applied.
    naive_win = net_impact["lift"] > 0
    net_neg = net_impact["classification"] == "net negative"
    summary["naive_win_but_net_negative"] = int((naive_win & net_neg).sum())
    summary["naive_wins"] = int(naive_win.sum())

    # SC-credibility downgrade stress test (additive; the headline split above is unchanged).
    # Any SC-chosen event flagged sc_low_credibility is downgraded to inconclusive, and the
    # resulting split is reported alongside the headline so both views are visible.
    downgrade = (net_impact["chosen_method"] == "SC") & net_impact["sc_low_credibility"]
    downgraded = net_impact["classification"].mask(downgrade, "inconclusive")
    summary["sc_low_credibility_downgrade"] = {
        "n_sc_low_credibility": int((net_impact["chosen_method"] == "SC").mul(
            net_impact["sc_low_credibility"]).sum()),
        "n_labels_changed": int((downgraded != net_impact["classification"]).sum()),
        "overall_split": _split_counts(downgraded),
        "note": ("Downgrades SC-chosen events whose promo lift is not separable from donor "
                 "placebo noise (sc_low_credibility) to inconclusive. Reported as a stress test; "
                 "the headline overall_split does NOT apply this downgrade."),
    }

    # Credibility-tiered breakdown of the confident labels (primary labels unchanged). A label is
    # high-confidence when its chosen estimate is itself credible (credible SC, or good-fit ITS),
    # lower-confidence otherwise (placebo-low-credibility SC, or poor-fit ITS).
    tiers: dict[str, object] = {}
    for label in ("net negative", "net positive"):
        sub = net_impact[net_impact["classification"] == label]
        tiers[label] = {
            "total": int(len(sub)),
            "high_confidence": int(sub["estimate_credible"].sum()),
            "lower_confidence": int((~sub["estimate_credible"]).sum()),
        }
    tiers["note"] = ("high_confidence = primary label held AND the chosen estimate is credible "
                     "(chosen SC with sc_low_credibility == False, or chosen ITS with "
                     "poor_fit == False); lower_confidence otherwise. Primary labels are "
                     "unchanged; this is a reported stratification only.")
    summary["credibility_tiers"] = tiers

    # Placebo-calibrated robustness: among LOWER-CONFIDENCE net-negative events, add the optimistic
    # placebo net noise (90th-percentile, then max, in the direction that helps the promotion) to the
    # net incremental units and count how many REMAIN net negative. This probes whether a
    # lower-confidence net-negative label survives pure donor placebo noise.
    gm_price = econ.gross_margin * econ.assumed_price
    lower_nn = net_impact[(net_impact["classification"] == "net negative")
                          & (~net_impact["estimate_credible"])].copy()
    survive_p90 = survive_max = 0
    for row in lower_nn.itertuples(index=False):
        shift_p90 = row.placebo_net_p90 if np.isfinite(row.placebo_net_p90) else 0.0
        shift_max = row.placebo_net_max if np.isfinite(row.placebo_net_max) else 0.0
        if net_profit_after_shift(row.net_incremental_units, shift_p90,
                                  row.discount_cost, gm_price) < 0:
            survive_p90 += 1
        if net_profit_after_shift(row.net_incremental_units, shift_max,
                                  row.discount_cost, gm_price) < 0:
            survive_max += 1
    summary["n_sc_low_credibility_all"] = int(frame["sc_low_credibility"].sum())
    summary["placebo_robustness"] = {
        "n_lower_conf_net_negative": int(len(lower_nn)),
        "survive_p90": int(survive_p90),
        "survive_max": int(survive_max),
        "note": ("Among lower-confidence net-negative events, how many stay net negative after "
                 "adding the 90th-percentile / maximum placebo net effect (optimistic direction) to "
                 "their net incremental units. Placebo net effects come from the in-space placebo "
                 "donors (synthetic_control.py)."),
    }

    # Break-even economics summary (dependence on the modeled discount/margin assumptions).
    summary["break_even"] = _break_even_summary(net_impact, cfg)

    # Sensitivity grid over the two modeled assumptions.
    summary["sensitivity"] = _sensitivity_grid(frame, cfg)

    return net_impact, summary


def _break_even_summary(net_impact: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Summarize per-event break-even figures and assumption-dependence counts."""
    be = cfg.break_even
    positive_lift = (net_impact["lift"] > 0) & (net_impact["net_incremental_units"] > 0)
    currently_not_positive = net_impact["net_profit_impact"] <= 0
    # Events that would turn net positive (point estimate) at a shallower discount / higher margin.
    flip_discount = (currently_not_positive
                     & (net_impact["net_incremental_units"] > 0)
                     & (net_impact["break_even_discount"] > be.alt_discount_depth))
    flip_margin = (currently_not_positive
                   & (net_impact["net_incremental_units"] > 0)
                   & (net_impact["break_even_margin"] < be.alt_gross_margin))
    return {
        "median_break_even_discount_positive_lift": (
            round(float(net_impact.loc[positive_lift, "break_even_discount"].median()), 4)
            if positive_lift.any() else None),
        "median_break_even_margin_positive_lift": (
            round(float(net_impact.loc[positive_lift, "break_even_margin"].median()), 4)
            if positive_lift.any() else None),
        "n_never_break_even": int(net_impact["never_breaks_even"].sum()),
        "alt_discount_depth": be.alt_discount_depth,
        "alt_gross_margin": be.alt_gross_margin,
        "n_flip_positive_at_alt_discount": int(flip_discount.sum()),
        "n_flip_positive_at_alt_margin": int(flip_margin.sum()),
        "note": ("break_even_discount = gross_margin x net_incremental_units / promo_units_sold; "
                 "break_even_margin = discount_depth x promo_units_sold / net_incremental_units. "
                 "Flip counts use point estimates: currently non-positive promotions that would "
                 "turn net positive at the alt discount/margin."),
    }


def _sensitivity_grid(frame: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Sweep discount_depth x gross_margin; report split + median net profit per cell."""
    econ = cfg.economics
    cells: list[dict[str, object]] = []
    for depth in cfg.sensitivity.discount_depths:
        for margin in cfg.sensitivity.gross_margins:
            discount_cost = depth * econ.assumed_price * frame["promo_units_sold"]
            gm_price = margin * econ.assumed_price
            net_profit = gm_price * frame["net_incremental_units"] - discount_cost
            net_profit_lo = gm_price * frame["net_units_lo"] - discount_cost
            net_profit_hi = gm_price * frame["net_units_hi"] - discount_cost
            cell = frame[["net_incremental_units", "net_units_lo", "net_units_hi",
                          "low_confidence"]].copy()
            cell["net_profit_impact"] = net_profit
            cell["net_profit_lo"] = net_profit_lo
            cell["net_profit_hi"] = net_profit_hi
            labels = _classify_frame(cell, cfg.classification.inconclusive_basis)
            split = _split_counts(labels)
            cells.append({
                "discount_depth": depth,
                "gross_margin": margin,
                "median_net_profit_impact": round(float(net_profit.median()), 2),
                "pct_positive": split["pct"]["net positive"],
                "pct_negative": split["pct"]["net negative"],
                "pct_inconclusive": split["pct"]["inconclusive"],
                "counts": split["counts"],
            })
    return {
        "discount_depths": cfg.sensitivity.discount_depths,
        "gross_margins": cfg.sensitivity.gross_margins,
        "cells": cells,
    }


def run_net_impact(cfg: Config) -> pd.DataFrame:
    """Compute net impact, write the per-event table and the summary JSON."""
    net_impact, summary = build_net_impact(cfg)
    out = cfg.paths.data_processed
    out.mkdir(parents=True, exist_ok=True)
    net_impact.to_parquet(out / "net_impact.parquet", index=False)
    with (out / "net_impact_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    split = summary["overall_split"]
    logger.info("Net impact complete: %d events | split pos/neg/inconc = %s (%s%%)",
                split["total"], split["counts"], split["pct"])
    logger.info("Net profit impact: median %.1f | mean %.1f margin-units",
                summary["median_net_profit_impact"], summary["mean_net_profit_impact"])
    logger.info("Chosen estimate: %s | %d low-confidence forced inconclusive | %d naive "
                "wins, of which %d are net negative after cannibalization + discount",
                summary["chosen_method_counts"], summary["n_low_confidence"],
                summary["naive_wins"], summary["naive_win_but_net_negative"])
    be = summary["break_even"]
    logger.info("Break-even: median break-even discount (positive-lift events)=%s | "
                "%d never break even | %d flip positive at discount %.2f | %d at margin %.2f",
                be["median_break_even_discount_positive_lift"], be["n_never_break_even"],
                be["n_flip_positive_at_alt_discount"], be["alt_discount_depth"],
                be["n_flip_positive_at_alt_margin"], be["alt_gross_margin"])
    dg = summary["sc_low_credibility_downgrade"]
    logger.info("SC low-credibility downgrade stress test: %d SC events flagged, %d headline "
                "labels would change -> split %s", dg["n_sc_low_credibility"],
                dg["n_labels_changed"], dg["overall_split"]["counts"])
    tiers = summary["credibility_tiers"]
    logger.info("Credibility tiers: net negative %d high / %d lower | net positive %d high / "
                "%d lower", tiers["net negative"]["high_confidence"],
                tiers["net negative"]["lower_confidence"],
                tiers["net positive"]["high_confidence"],
                tiers["net positive"]["lower_confidence"])
    pr = summary["placebo_robustness"]
    logger.info("Placebo robustness (lower-confidence net negatives, n=%d): %d survive +p90 "
                "placebo noise, %d survive +max placebo noise", pr["n_lower_conf_net_negative"],
                pr["survive_p90"], pr["survive_max"])
    return net_impact


def main() -> None:
    """CLI entry point (part of the ``impact`` Makefile target)."""
    parser = argparse.ArgumentParser(description="Compute net profit impact per promotion.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    run_net_impact(cfg)


if __name__ == "__main__":
    main()
