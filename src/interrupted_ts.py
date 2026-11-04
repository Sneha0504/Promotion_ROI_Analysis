"""Unit 2, step 2: interrupted time series counterfactuals (PLAN sec. 6a).

For every analyzable event an ordinary-least-squares model is fit on the **pre-period
only** and projected forward through the promotion and post-period as the counterfactual
"what sales would have been without the promotion". The model regresses ``log1p`` unit
sales on a linear time trend, day-of-week dummies, a store-specific holiday indicator and
(optionally) the daily oil price. The promotion effect is the back-transformed
``actual - counterfactual`` summed over the promotion window and, separately, the
post-period, with a confidence interval from a moving-block residual bootstrap.

Design choices (documented)
---------------------------
* **log1p transform** stabilizes the multiplicative, heavy-tailed daily demand and keeps
  the counterfactual non-negative. Back-transformation to original units uses Duan's
  smearing estimator (multiply by ``mean(exp(residual))``) to correct retransformation
  bias, so the counterfactual estimates the mean, not the median.
* **Closure days are masked**: days the store did not trade (NaN in the panel) are dropped
  from fitting and excluded from every effect sum.
* **OLS is the primary method.** Residual autocorrelation is measured (Durbin-Watson and
  Ljung-Box) and surfaced per event; a SARIMA upgrade is the documented fallback for
  severely autocorrelated series (``autocorr_flag``), but OLS with a block bootstrap that
  respects weekly dependence is used throughout for comparability.
* **Poor pre-fit events are flagged** (``poor_fit``) when the pre-period R^2 is below
  ``its.min_pre_r2`` or the normalized pre-period RMSE exceeds ``its.max_pre_nrmse``.
* **Implausible counterfactuals are guarded** by two criteria: the back-transformed log-linear
  trend can compound exponentially when projected out-of-sample. A projected daily counterfactual
  above ``its.cf_cap_multiple`` x the pre-period **max** daily sales, **or** above
  ``its.cf_cap_mean_multiple`` x the pre-period **mean** daily sales, is flagged
  (``cf_implausible``, which also sets ``poor_fit``); the stored counterfactual is capped at the
  binding (smaller) ceiling. The max-based criterion catches blow-ups above the historical peak;
  the mean-based criterion catches milder artifacts that stay under the max cap but still balloon
  far above the no-promotion baseline level. The **same ceiling is applied to every bootstrap
  draw**, so the effect CI cannot explode beyond the capped point estimate; the reported interval
  is also guaranteed to contain the point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

from src.config import Config, load_config
from src.features import HolidayCalendar, load_oil
from src.logging_utils import get_logger

logger = get_logger(__name__)


def _design_matrix(
    calendar: pd.DatetimeIndex,
    holiday: np.ndarray,
    oil: pd.Series | None,
) -> tuple[np.ndarray, list[str]]:
    """Build the OLS design matrix (with intercept) over the full event calendar.

    Columns: intercept, linear trend, six day-of-week dummies (Monday as the baseline), a
    holiday indicator and, when enabled, the forward-filled daily oil price.
    """
    n = len(calendar)
    trend = np.arange(n, dtype=float)
    dow = calendar.dayofweek.to_numpy()
    dummies = np.column_stack([(dow == d).astype(float) for d in range(1, 7)])

    cols = [np.ones(n), trend, *dummies.T, holiday.astype(float)]
    names = ["const", "trend", *[f"dow_{d}" for d in range(1, 7)], "holiday"]
    if oil is not None:
        oil_vals = oil.reindex(calendar).to_numpy()
        cols.append(oil_vals)
        names.append("oil")
    return np.column_stack(cols), names


def _moving_block(resid: np.ndarray, length: int, block: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Return a moving-block bootstrap resample of ``resid`` of the requested ``length``."""
    n = len(resid)
    if n == 0:
        return np.zeros(length)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    out = np.concatenate([resid[s:s + block] for s in starts])
    return out[:length]


def fit_event_its(
    target: pd.DataFrame,
    event: pd.Series,
    holiday_cal: HolidayCalendar,
    oil: pd.Series | None,
    cfg: Config,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit ITS for one event; return (result row, per-day display frame).

    ``target`` is the event's target-role panel rows (one per calendar day) with
    ``unit_sales`` (closure days as NaN) and ``segment`` labels.
    """
    its = cfg.its
    target = target.sort_values("date").reset_index(drop=True)
    calendar = pd.DatetimeIndex(target["date"])
    sales = target["unit_sales"].to_numpy(dtype=float)
    segment = target["segment"].to_numpy()

    holiday = holiday_cal.flags(calendar, event["state"], event["city"])
    design, names = _design_matrix(calendar, holiday, oil if its.use_oil else None)

    y_full = np.log1p(sales) if its.log_transform else sales.copy()

    pre = (segment == "pre") & ~np.isnan(sales)
    promo = (segment == "promo") & ~np.isnan(sales)
    post = (segment == "post") & ~np.isnan(sales)

    x_pre, y_pre = design[pre], y_full[pre]
    model = sm.OLS(y_pre, x_pre).fit()
    resid = y_pre - model.predict(x_pre)

    # Duan smearing factor for mean back-transformation (1.0 in the untransformed case).
    smear = float(np.mean(np.exp(resid))) if its.log_transform else 1.0

    def back_point(linpred: np.ndarray) -> np.ndarray:
        """Central counterfactual (mean back-transform with Duan smearing).

        For ``y = log1p(sales)`` the mean of ``sales`` is ``exp(linpred) * S - 1`` where
        ``S = mean(exp(resid))`` — i.e. ``expm1(linpred) * S + (S - 1)``, not
        ``expm1(linpred) * S`` (which would be ``exp(linpred) * S - S``).
        """
        if its.log_transform:
            return np.clip(np.expm1(linpred) * smear + (smear - 1.0), 0.0, None)
        return np.clip(linpred, 0.0, None)

    def back_boot(linpred: np.ndarray) -> np.ndarray:
        """Back-transform for bootstrap draws that ALREADY carry resampled residuals.

        Each draw adds an empirical log-space residual ``e``, so ``exp(linpred + e) - 1``
        is the plain inverse: averaging ``exp(e)`` over draws reproduces the smearing
        factor, so applying ``smear`` here too would double-count the retransformation.
        """
        if its.log_transform:
            return np.clip(np.expm1(linpred), 0.0, None)
        return np.clip(linpred, 0.0, None)

    linpred_full = design @ model.params
    cf_full = back_point(linpred_full)

    # Plausibility guard. A log-linear trend back-transforms to an exponential, which can
    # compound to absurd levels when projected across the promo/post horizon (e.g. a
    # counterfactual of millions of units against a few-thousand-unit actual). Any projected
    # daily counterfactual above ``cf_cap_multiple`` x the pre-period MAX daily sales is
    # treated as implausible: the event is flagged (``cf_implausible`` -> ``poor_fit``) and the
    # stored counterfactual is capped at that ceiling so the saved effects and the
    # display_series line stay bounded. For an event never flagged implausible the point
    # counterfactual stays below the ceiling, so its point effects are unchanged; the same
    # finite ceiling can still bound that event's bootstrap draws (see below).
    pre_sales = sales[pre]
    pre_max = float(np.nanmax(pre_sales)) if pre_sales.size else 0.0
    pre_mean_daily = float(np.nanmean(pre_sales)) if pre_sales.size else 0.0
    # Two plausibility ceilings; the binding one is the smaller. The MAX-based ceiling catches
    # projections that balloon above the highest pre-period day; the MEAN-based ceiling catches
    # milder artifacts that stay under the max cap but still sit far above the no-promotion
    # baseline level (a "no-promotion" counterfactual should sit near the pre-period mean).
    ceiling_max = its.cf_cap_multiple * pre_max if pre_max > 0 else np.inf
    ceiling_mean = its.cf_cap_mean_multiple * pre_mean_daily if pre_mean_daily > 0 else np.inf
    ceiling = min(ceiling_max, ceiling_mean)
    proj_mask = promo | post
    cf_implausible = bool(np.isfinite(ceiling) and proj_mask.any()
                          and np.any(cf_full[proj_mask] > ceiling))
    if np.isfinite(ceiling):
        cf_full = np.minimum(cf_full, ceiling)

    # Point effects over open days in each segment.
    def seg_effect(mask: np.ndarray) -> tuple[float, float, float]:
        actual_sum = float(np.nansum(sales[mask]))
        cf_sum = float(np.sum(cf_full[mask]))
        return actual_sum - cf_sum, actual_sum, cf_sum

    promo_effect, promo_actual, promo_cf = seg_effect(promo)
    post_effect, post_actual, post_cf = seg_effect(post)

    # Moving-block residual bootstrap for the effect CI (captures parameter uncertainty
    # via refit on resampled pre-period residuals plus predictive noise on the horizon).
    draws = its.bootstrap_draws
    promo_draws = np.empty(draws)
    post_draws = np.empty(draws)
    yhat_pre = model.predict(x_pre)
    xtx_inv_xt = np.linalg.pinv(x_pre)  # (X'X)^-1 X' for fast refits
    x_promo, x_post = design[promo], design[post]
    for b in range(draws):
        y_star = yhat_pre + _moving_block(resid, len(resid), its.block_size, rng)
        beta_star = xtx_inv_xt @ y_star
        noise_promo = _moving_block(resid, promo.sum(), its.block_size, rng)
        noise_post = _moving_block(resid, post.sum(), its.block_size, rng)
        # Apply the SAME per-day plausibility ceiling to each bootstrap draw's counterfactual
        # as to the point estimate, so an exploding log-linear projection cannot inflate the CI
        # beyond the capped point. The ceiling is finite whenever the pre-period has positive
        # sales, so it can also narrow the interval of an event that was NOT flagged implausible
        # (its point stayed under the ceiling) but whose draws exceed it; this only bounds a
        # runaway interval and never widens a sound one. It is a true no-op only when the ceiling
        # is infinite (zero pre-period sales) or no draw exceeds it.
        cf_promo_b = np.minimum(back_boot(x_promo @ beta_star + noise_promo), ceiling)
        cf_post_b = np.minimum(back_boot(x_post @ beta_star + noise_post), ceiling)
        promo_draws[b] = promo_actual - float(cf_promo_b.sum())
        post_draws[b] = post_actual - float(cf_post_b.sum())

    alpha = 1.0 - cfg.significance.ci_level
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    # Percentile-bootstrap bounds, then guarantee the invariant that the point estimate lies
    # within its own reported interval (a percentile CI can otherwise narrowly exclude the
    # model point estimate, which is not the bootstrap median). Side effect: for a few capped
    # events a reported bound equals the point estimate exactly (the clamp supplied it), which
    # is a consistency guard, not a genuine interval endpoint.
    promo_lo = min(float(np.percentile(promo_draws, lo)), promo_effect)
    promo_hi = max(float(np.percentile(promo_draws, hi)), promo_effect)
    post_lo = min(float(np.percentile(post_draws, lo)), post_effect)
    post_hi = max(float(np.percentile(post_draws, hi)), post_effect)

    # Diagnostics (log-space residuals). Uses the capped counterfactual for consistency; on
    # plausible events (the pre-period fit is in range) the cap does not touch the pre window.
    cf_pre = cf_full[pre]
    actual_pre = sales[pre]
    pre_rmse = float(np.sqrt(np.mean((actual_pre - cf_pre) ** 2)))
    pre_mean = float(np.mean(actual_pre)) if actual_pre.size else np.nan
    pre_nrmse = pre_rmse / pre_mean if pre_mean else np.nan
    dw = float(durbin_watson(resid))
    lb_lags = min(its.ljung_box_lags, max(1, len(resid) // 5))
    try:
        lb_p = float(acorr_ljungbox(resid, lags=[lb_lags],
                                    return_df=True)["lb_pvalue"].iloc[0])
    except (ValueError, np.linalg.LinAlgError):
        lb_p = np.nan

    poor_fit = bool(model.rsquared < its.min_pre_r2
                    or (not np.isnan(pre_nrmse) and pre_nrmse > its.max_pre_nrmse)
                    or cf_implausible)
    autocorr_flag = bool((not np.isnan(lb_p) and lb_p < 0.05) or dw < 1.0 or dw > 3.0)

    result = {
        "event_id": event["event_id"],
        "method": "OLS",
        "n_pre_obs": int(pre.sum()),
        "pre_r2": float(model.rsquared),
        "pre_rmse": pre_rmse,
        "pre_nrmse": float(pre_nrmse) if not np.isnan(pre_nrmse) else np.nan,
        "durbin_watson": dw,
        "ljung_box_p": lb_p,
        "autocorr_flag": autocorr_flag,
        "cf_implausible": cf_implausible,
        "poor_fit": poor_fit,
        "smearing": smear,
        "promo_actual": promo_actual,
        "promo_cf": promo_cf,
        "promo_effect": promo_effect,
        "promo_effect_lo": promo_lo,
        "promo_effect_hi": promo_hi,
        "post_actual": post_actual,
        "post_cf": post_cf,
        "post_effect": post_effect,
        "post_effect_lo": post_lo,
        "post_effect_hi": post_hi,
    }

    display = pd.DataFrame({
        "event_id": event["event_id"],
        "date": calendar,
        "actual": sales,
        "its_cf": np.where(np.isnan(sales), np.nan, cf_full),
        "segment": segment,
        "in_promo": segment == "promo",
    })
    return result, display


def run_its(cfg: Config, limit: int | None = None) -> pd.DataFrame:
    """Fit ITS for every analyzable event; write results + a per-day display frame."""
    panels = pd.read_parquet(cfg.paths.data_interim / "analysis_panels.parquet")
    panels["date"] = pd.to_datetime(panels["date"])
    events = pd.read_parquet(cfg.paths.data_processed / "analyzable_events.parquet")
    if limit is not None:
        events = events.head(limit).copy()

    holiday_cal = HolidayCalendar(cfg)
    oil = load_oil(cfg) if cfg.its.use_oil else None
    rng = np.random.default_rng(cfg.random_seed)

    targets = panels[panels["role"] == "target"]
    target_by_event = {eid: g for eid, g in targets.groupby("event_id", sort=False)}

    results: list[dict[str, object]] = []
    displays: list[pd.DataFrame] = []
    for i, (_, event) in enumerate(events.iterrows(), start=1):
        target = target_by_event.get(event["event_id"])
        if target is None:
            continue
        result, display = fit_event_its(target, event, holiday_cal, oil, cfg, rng)
        results.append(result)
        displays.append(display)
        if i % 50 == 0 or i == len(events):
            logger.info("Fit ITS for %d/%d events", i, len(events))

    results_df = pd.DataFrame(results)
    display_df = pd.concat(displays, ignore_index=True)

    out = cfg.paths.data_processed
    out.mkdir(parents=True, exist_ok=True)
    cfg.paths.data_interim.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(out / "its_results.parquet", index=False)
    # Per-day ITS counterfactual is an intermediate; the synth step merges it into the
    # committed display_series.parquet, so it lives in (git-ignored) data/interim.
    display_df.to_parquet(cfg.paths.data_interim / "its_display.parquet", index=False)

    n_poor = int(results_df["poor_fit"].sum())
    n_autocorr = int(results_df["autocorr_flag"].sum())
    n_implausible = int(results_df["cf_implausible"].sum())
    logger.info("ITS complete: %d events, %d flagged poor-fit (%d of them for implausible "
                "counterfactuals), %d flagged autocorrelated",
                len(results_df), n_poor, n_implausible, n_autocorr)
    logger.info("Median pre-period R^2: %.3f | median |promo effect|: %.1f units",
                float(results_df["pre_r2"].median()),
                float(results_df["promo_effect"].abs().median()))
    return results_df


def main() -> None:
    """CLI entry point for the ``its`` Makefile target."""
    parser = argparse.ArgumentParser(description="Fit interrupted time series models.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    run_its(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
