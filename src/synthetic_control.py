"""Unit 2, step 3: synthetic control counterfactuals (PLAN sec. 6b).

For every analyzable event a synthetic control is built as a weighted average of the
event's donor item-stores, with weights **non-negative and summing to one** (the standard
synthetic-control simplex constraint). Weights are fit to the target's **pre-period** daily
sales by constrained least squares (SciPy SLSQP over the simplex, with an NNLS-normalized
fallback). The counterfactual is the donor-weighted series carried through the promotion
and post-period; the effect is ``actual - synthetic`` summed over each window.

Credibility and comparison
--------------------------
* The **pre-period fit is reported explicitly** as the normalized pre-period RMSPE
  (root-mean-square prediction error divided by mean pre-period sales). Poor pre-fit
  (``> synth.max_pre_rmspe``) is flagged -- a synthetic control that cannot reproduce the
  pre-period is not trustworthy in the post-period.
* **In-space placebo inference** (Abadie) is reported as an additive credibility layer: each
  donor is re-fit as a pseudo-treated unit on the remaining donors, and the real effect's
  placebo p-value is the rank of its RMSPE-normalized effect among all units. An event whose
  promo-window placebo p-value exceeds ``synth.placebo_pvalue_threshold`` is flagged
  ``sc_low_credibility`` (its lift is not distinguishable from donor placebo noise). This does
  **not** alter the point effects or the headline pipeline.
* Effects are then **compared with the interrupted-time-series estimates** for the same
  events (correlation, sign agreement, divergence), and a merged per-day display series
  (actual, ITS counterfactual, SC counterfactual) is written for the dashboard.

Operates in original units (no log transform): synthetic control matches levels directly.
Closure days (NaN in the panel) are dropped from fitting and excluded from every sum.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from src.config import Config, load_config
from src.logging_utils import get_logger

logger = get_logger(__name__)


def solve_weights(donor_pre: np.ndarray, target_pre: np.ndarray,
                  max_iter: int) -> tuple[np.ndarray, str]:
    """Solve for simplex weights minimizing ``||target_pre - donor_pre @ w||^2``.

    Weights are constrained non-negative and to sum to one. The least-squares objective on
    raw grocery sales is O(1e4-1e6), on which SLSQP stalls; the problem is therefore
    **scaled** (target and donors divided by ``max|target|``) so the solver sees an O(1)
    problem. Several candidate solutions are then compared by the *actual* objective and the
    best is returned -- the uniform-weights baseline is always a candidate, so the result is
    **never worse than uniform**. (A normalized-NNLS point is not the simplex optimum, so it
    is only ever accepted when it genuinely beats the SLSQP solution and uniform.)

    Args:
        donor_pre: pre-period donor matrix, shape ``(n_days, n_donors)``.
        target_pre: pre-period target vector, shape ``(n_days,)``.
        max_iter: SLSQP iteration cap.

    Returns:
        ``(weights, method)`` where ``method`` is the winning candidate
        (``'slsqp'`` | ``'nnls'`` | ``'uniform'``).
    """
    n_donors = donor_pre.shape[1]
    uniform = np.full(n_donors, 1.0 / n_donors)

    scale = float(np.max(np.abs(target_pre)))
    if scale <= 0:
        scale = 1.0
    a = donor_pre / scale
    b = target_pre / scale

    def objective(w: np.ndarray) -> float:
        resid = b - a @ w
        return float(resid @ resid)

    def grad(w: np.ndarray) -> np.ndarray:
        resid = b - a @ w
        return -2.0 * a.T @ resid

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                    "jac": lambda w: np.ones_like(w)},)
    bounds = [(0.0, 1.0)] * n_donors

    candidates: list[tuple[str, np.ndarray]] = [("uniform", uniform)]

    res = minimize(objective, uniform, jac=grad, bounds=bounds, constraints=constraints,
                   method="SLSQP", options={"maxiter": max_iter, "ftol": 1e-12})
    if res.x is not None and np.all(np.isfinite(res.x)):
        w = np.clip(res.x, 0.0, None)
        total = w.sum()
        if total > 0:
            candidates.append(("slsqp", w / total))

    try:
        coef, _ = nnls(a, b)
        total = coef.sum()
        if total > 0:
            candidates.append(("nnls", coef / total))
    except (ValueError, RuntimeError):
        pass

    method, weights = min(candidates, key=lambda kv: objective(kv[1]))
    return weights, method


def _moving_block(resid: np.ndarray, length: int, block: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap resample of ``resid`` to the requested ``length``."""
    n = len(resid)
    if n == 0 or length == 0:
        return np.zeros(length)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return np.concatenate([resid[s:s + block] for s in starts])[:length]


def _sc_point_fit(
    y: np.ndarray, donor_vals: np.ndarray, pre: np.ndarray, promo: np.ndarray,
    post: np.ndarray, syn: object,
) -> tuple[float, float, float] | None:
    """Point-only SC fit used for placebo inference (no bootstrap).

    Mirrors the real fit's weight-solve exactly (pre-period smoothing + level fixed effect) and
    returns ``(promo_effect, post_effect, pre_rmspe)``, or None if under-determined. Kept separate
    from :func:`fit_event_synth` so the headline path is untouched.
    """
    if donor_vals.shape[1] == 0:
        return None
    donor_ok = ~np.isnan(donor_vals).any(axis=1)
    fit_rows = pre & ~np.isnan(y) & donor_ok
    if fit_rows.sum() < donor_vals.shape[1] + 1:
        return None
    y_pre = y[fit_rows]
    d_pre = donor_vals[fit_rows]
    if syn.smooth_window > 1:
        y_fit = pd.Series(y_pre).rolling(syn.smooth_window, min_periods=1,
                                         center=True).mean().to_numpy()
        d_fit = pd.DataFrame(d_pre).rolling(syn.smooth_window, min_periods=1,
                                            center=True).mean().to_numpy()
    else:
        y_fit, d_fit = y_pre, d_pre
    if syn.allow_intercept:
        mu_t = float(np.mean(y_pre))
        mu_d = d_pre.mean(axis=0)
    else:
        mu_t = 0.0
        mu_d = np.zeros(donor_vals.shape[1])
    weights, _ = solve_weights(d_fit - mu_d, y_fit - mu_t, syn.solver_max_iter)
    synth_full = mu_t + (donor_vals - mu_d) @ weights
    pre_rmspe = float(np.sqrt(np.mean((y_pre - synth_full[fit_rows]) ** 2)))

    def seg(mask: np.ndarray) -> float:
        rows = mask & ~np.isnan(y) & donor_ok
        return float(np.sum(y[rows]) - np.sum(synth_full[rows]))

    return seg(promo), seg(post), pre_rmspe


def placebo_pvalues(
    donor_vals: np.ndarray, pre: np.ndarray, promo: np.ndarray, post: np.ndarray,
    syn: object, real_promo: float, real_post: float, real_pre_rmspe: float,
) -> tuple[float, float, int, float, float]:
    """In-space placebo inference (Abadie): p-values + placebo net-effect optimistic tail.

    Each donor is re-fit as a *pseudo-treated* unit against the remaining donors; its placebo
    effect is normalized by its own pre-period RMSPE (the standard ratio that corrects for fit
    quality). The two-sided p-value is the rank of the real unit's ratio among all units:
    ``(1 + #{placebo ratio >= real ratio}) / (1 + n_placebos)``.

    The same placebos also give a distribution of placebo **net** effects (promo + post, in units)
    for untreated donors. Its **90th percentile** (``placebo_net_p90``) and **maximum**
    (``placebo_net_max``) are the optimistic-side scale of pure placebo noise, reused downstream to
    stress-test net-negative labels (how much "free" lift a non-promotion could spuriously show).

    Returns ``(p_promo, p_post, n_placebos, placebo_net_p90, placebo_net_max)``.
    """
    eps = 1e-9
    real_ratio_promo = abs(real_promo) / max(real_pre_rmspe, eps)
    real_ratio_post = abs(real_post) / max(real_pre_rmspe, eps)
    ge_promo = ge_post = n = 0
    net_effects: list[float] = []
    for j in range(donor_vals.shape[1]):
        res = _sc_point_fit(donor_vals[:, j], np.delete(donor_vals, j, axis=1),
                            pre, promo, post, syn)
        if res is None:
            continue
        pe, po, rmspe = res
        n += 1
        net_effects.append(pe + po)
        if abs(pe) / max(rmspe, eps) >= real_ratio_promo:
            ge_promo += 1
        if abs(po) / max(rmspe, eps) >= real_ratio_post:
            ge_post += 1
    if n == 0:
        return float("nan"), float("nan"), 0, float("nan"), float("nan")
    net_arr = np.asarray(net_effects, dtype=float)
    return ((1 + ge_promo) / (1 + n), (1 + ge_post) / (1 + n), n,
            float(np.percentile(net_arr, 90)), float(np.max(net_arr)))


def fit_event_synth(
    target: pd.DataFrame,
    donors: pd.DataFrame,
    event_id: str,
    cfg: Config,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame] | None:
    """Fit synthetic control for one event; return (result row, display frame) or None."""
    syn = cfg.synth
    target = target.sort_values("date")
    calendar = pd.DatetimeIndex(target["date"])
    y = target.set_index("date")["unit_sales"].reindex(calendar).to_numpy(dtype=float)
    segment = target.set_index("date")["segment"].reindex(calendar).to_numpy()

    donor_mat = donors.pivot_table(index="date", columns="donor_rank",
                                   values="unit_sales", aggfunc="first")
    donor_mat = donor_mat.reindex(calendar)
    donor_vals = donor_mat.to_numpy(dtype=float)
    if donor_vals.shape[1] == 0:
        return None

    pre = (segment == "pre")
    promo = (segment == "promo")
    post = (segment == "post")

    # Rows usable for fitting: target and every donor observed (no closure NaN).
    donor_ok = ~np.isnan(donor_vals).any(axis=1)
    fit_rows = pre & ~np.isnan(y) & donor_ok
    if fit_rows.sum() < donor_vals.shape[1] + 1:
        return None  # under-determined; treated as a drop at this stage

    # Fit weights on the de-noised weekly signal: daily item sales are Poisson-like and
    # far too noisy to match day-by-day, so a rolling mean over the pre-period (computed
    # within the pre-period only, no promo leakage) lets the control track the trend and
    # weekly pattern. Effects below are still computed on RAW daily sales.
    y_pre = y[fit_rows]
    d_pre = donor_vals[fit_rows]
    if syn.smooth_window > 1:
        y_fit = pd.Series(y_pre).rolling(syn.smooth_window, min_periods=1,
                                         center=True).mean().to_numpy()
        d_fit = (pd.DataFrame(d_pre).rolling(syn.smooth_window, min_periods=1,
                                             center=True).mean().to_numpy())
    else:
        y_fit, d_fit = y_pre, d_pre

    # Pre-period level fixed effect: demean before fitting so weights capture co-movement,
    # then add the target's pre-mean back into the counterfactual. Weights remain on the
    # simplex; the synthetic's pre-mean equals the target's pre-mean by construction.
    if syn.allow_intercept:
        mu_t = float(np.mean(y_pre))
        mu_d = d_pre.mean(axis=0)
    else:
        mu_t = 0.0
        mu_d = np.zeros(donor_vals.shape[1])

    weights, solver_method = solve_weights(d_fit - mu_d, y_fit - mu_t, syn.solver_max_iter)
    synth_full = mu_t + (donor_vals - mu_d) @ weights

    # Pre-period credibility: RMSPE on the smoothed signal (the fit target) is the headline
    # flag; the raw daily RMSPE is also reported for transparency.
    pre_resid = y_pre - synth_full[fit_rows]
    pre_rmspe = float(np.sqrt(np.mean(pre_resid ** 2)))
    pre_mean = float(np.mean(y_pre))
    norm_rmspe = pre_rmspe / pre_mean if pre_mean else np.nan
    smooth_resid = y_fit - (mu_t + (d_fit - mu_d) @ weights)
    smooth_rmspe = float(np.sqrt(np.mean(smooth_resid ** 2)))
    norm_smooth_rmspe = smooth_rmspe / pre_mean if pre_mean else np.nan
    poor_fit = bool(not np.isnan(norm_smooth_rmspe)
                    and norm_smooth_rmspe > syn.max_pre_rmspe)

    def seg_sums(mask: np.ndarray) -> tuple[float, float, float, int]:
        rows = mask & ~np.isnan(y) & donor_ok
        actual_sum = float(np.sum(y[rows]))
        synth_sum = float(np.sum(synth_full[rows]))
        return actual_sum - synth_sum, actual_sum, synth_sum, int(rows.sum())

    promo_effect, promo_actual, promo_synth, n_promo = seg_sums(promo)
    post_effect, post_actual, post_synth, n_post = seg_sums(post)

    # In-space placebo inference (additive credibility layer; does not alter the point effects).
    p_promo, p_post, n_placebos, placebo_net_p90, placebo_net_max = placebo_pvalues(
        donor_vals, pre, promo, post, syn, promo_effect, post_effect, pre_rmspe)
    sc_low_credibility = bool(np.isnan(p_promo) or p_promo > syn.placebo_pvalue_threshold)

    # Residual-bootstrap CI (adds resampled pre-period gaps to the synthetic horizon).
    draws = syn.bootstrap_draws
    promo_draws = np.empty(draws)
    post_draws = np.empty(draws)
    for b in range(draws):
        promo_draws[b] = promo_actual - (promo_synth
                                         + _moving_block(pre_resid, n_promo,
                                                         syn.block_size, rng).sum())
        post_draws[b] = post_actual - (post_synth
                                       + _moving_block(pre_resid, n_post,
                                                       syn.block_size, rng).sum())
    alpha = 1.0 - cfg.significance.ci_level
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    result = {
        "event_id": event_id,
        "n_donors": int(donor_vals.shape[1]),
        "n_pre_obs": int(fit_rows.sum()),
        "pre_rmspe": pre_rmspe,
        "norm_pre_rmspe": float(norm_rmspe) if not np.isnan(norm_rmspe) else np.nan,
        "norm_smooth_rmspe": (float(norm_smooth_rmspe)
                              if not np.isnan(norm_smooth_rmspe) else np.nan),
        "solver_method": solver_method,
        "max_weight": float(weights.max()),
        "effective_donors": float(1.0 / np.sum(weights ** 2)) if np.any(weights) else 0.0,
        "poor_fit": poor_fit,
        "promo_actual": promo_actual,
        "promo_synth": promo_synth,
        "promo_effect": promo_effect,
        "promo_effect_lo": float(np.percentile(promo_draws, lo)),
        "promo_effect_hi": float(np.percentile(promo_draws, hi)),
        "post_actual": post_actual,
        "post_synth": post_synth,
        "post_effect": post_effect,
        "post_effect_lo": float(np.percentile(post_draws, lo)),
        "post_effect_hi": float(np.percentile(post_draws, hi)),
        "placebo_pvalue_promo": float(p_promo) if not np.isnan(p_promo) else np.nan,
        "placebo_pvalue_post": float(p_post) if not np.isnan(p_post) else np.nan,
        "n_placebos": int(n_placebos),
        "sc_low_credibility": sc_low_credibility,
        "placebo_net_p90": placebo_net_p90,
        "placebo_net_max": placebo_net_max,
    }

    # Display counterfactual: masked where target or any donor is unobserved.
    sc_cf = np.where(np.isnan(y) | ~donor_ok, np.nan, synth_full)
    display = pd.DataFrame({"event_id": event_id, "date": calendar, "sc_cf": sc_cf})
    return result, display


def build_comparison(its_results: pd.DataFrame, synth_results: pd.DataFrame) -> pd.DataFrame:
    """Merge ITS and SC effects and tabulate per-event agreement."""
    its = its_results[["event_id", "promo_effect", "post_effect", "poor_fit"]].rename(
        columns={"promo_effect": "its_promo_effect", "post_effect": "its_post_effect",
                 "poor_fit": "its_poor_fit"})
    syn = synth_results[["event_id", "promo_effect", "post_effect", "poor_fit"]].rename(
        columns={"promo_effect": "sc_promo_effect", "post_effect": "sc_post_effect",
                 "poor_fit": "sc_poor_fit"})
    merged = its.merge(syn, on="event_id", how="inner")
    merged["promo_sign_agree"] = (np.sign(merged["its_promo_effect"])
                                  == np.sign(merged["sc_promo_effect"]))
    merged["post_sign_agree"] = (np.sign(merged["its_post_effect"])
                                 == np.sign(merged["sc_post_effect"]))
    merged["promo_abs_diff"] = (merged["its_promo_effect"]
                                - merged["sc_promo_effect"]).abs()
    merged["both_trustworthy"] = ~(merged["its_poor_fit"] | merged["sc_poor_fit"])
    return merged


def run_synth(cfg: Config, limit: int | None = None) -> pd.DataFrame:
    """Fit SC for every analyzable event; write results, comparison and display series."""
    panels = pd.read_parquet(cfg.paths.data_interim / "analysis_panels.parquet")
    panels["date"] = pd.to_datetime(panels["date"])
    events = pd.read_parquet(cfg.paths.data_processed / "analyzable_events.parquet")
    if limit is not None:
        events = events.head(limit).copy()

    rng = np.random.default_rng(cfg.random_seed)
    by_event = {eid: g for eid, g in panels.groupby("event_id", sort=False)}

    results: list[dict[str, object]] = []
    displays: list[pd.DataFrame] = []
    skipped = 0
    for i, (_, event) in enumerate(events.iterrows(), start=1):
        grp = by_event.get(event["event_id"])
        if grp is None:
            continue
        target = grp[grp["role"] == "target"]
        donors = grp[grp["role"] == "donor"]
        fit = fit_event_synth(target, donors, event["event_id"], cfg, rng)
        if fit is None:
            skipped += 1
            continue
        result, display = fit
        results.append(result)
        displays.append(display)
        if i % 50 == 0 or i == len(events):
            logger.info("Fit SC for %d/%d events", i, len(events))

    synth_results = pd.DataFrame(results)
    sc_display = pd.concat(displays, ignore_index=True)

    out = cfg.paths.data_processed
    out.mkdir(parents=True, exist_ok=True)
    synth_results.to_parquet(out / "synth_results.parquet", index=False)

    # Comparison with ITS.
    its_results = pd.read_parquet(out / "its_results.parquet")
    comparison = build_comparison(its_results, synth_results)
    comparison.to_parquet(out / "effect_comparison.parquet", index=False)

    # Merged per-day display series (actual, ITS cf, SC cf) for the dashboard.
    its_display = pd.read_parquet(cfg.paths.data_interim / "its_display.parquet")
    display_series = its_display.merge(sc_display, on=["event_id", "date"], how="left")
    display_series = display_series.merge(
        events[["event_id", "promo_start", "promo_end"]], on="event_id", how="left")
    display_series.to_parquet(out / "display_series.parquet", index=False)

    solver_counts = synth_results["solver_method"].value_counts().to_dict()
    logger.info("SC solver outcomes (events by winning method): %s", solver_counts)
    n_poor = int(synth_results["poor_fit"].sum())
    n_low_cred = int(synth_results["sc_low_credibility"].sum())
    median_p = float(synth_results["placebo_pvalue_promo"].median(skipna=True))
    logger.info("SC placebo inference: median promo p-value=%.3f | %d events flagged "
                "sc_low_credibility (p > %.2f)", median_p, n_low_cred,
                cfg.synth.placebo_pvalue_threshold)
    trust = comparison[comparison["both_trustworthy"]]
    corr_promo = float(trust["its_promo_effect"].corr(trust["sc_promo_effect"]))
    sign_agree = float(comparison["promo_sign_agree"].mean())
    logger.info("SC complete: %d events (%d skipped), %d flagged poor-fit",
                len(synth_results), skipped, n_poor)
    logger.info("ITS vs SC (trustworthy n=%d): promo-effect corr=%.3f | "
                "promo sign agreement=%.1f%%", len(trust), corr_promo, 100 * sign_agree)
    return synth_results


def main() -> None:
    """CLI entry point for the ``synth`` Makefile target."""
    parser = argparse.ArgumentParser(description="Fit synthetic control models.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    run_synth(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
