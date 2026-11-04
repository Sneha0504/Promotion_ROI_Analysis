"""Unit tests for the Unit 2 panel + model building blocks.

Covered:
* ``dense_series`` fills absent *open* days with 0 demand and masks *closure* days as NaN
  (the carried-forward sparse-storage / store-closure rule), and clips returns to 0;
* ``solve_weights`` returns simplex weights (non-negative, summing to 1) that are never
  worse than uniform weights by the least-squares objective;
* the ITS promotion/post effects equal the summed ``actual - counterfactual`` over the
  corresponding window on a constructed series.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.build_panel import dense_series
from src.config import load_config
from src.interrupted_ts import fit_event_its
from src.synthetic_control import fit_event_synth, solve_weights


def test_dense_series_open_zero_and_closure_nan() -> None:
    """Absent open days become 0, closure days become NaN, and returns clip to 0."""
    calendar = pd.date_range("2015-01-01", periods=5, freq="D")
    # Store 1 traded on all days except 2015-01-03 (a closure).
    open_days = {1: {calendar[0], calendar[1], calendar[3], calendar[4]}}
    raw = pd.DataFrame({
        "item_nbr": [100, 100, 100],
        "unit_sales": [5.0, -2.0, 7.0],      # the -2 is a return -> clip to 0
        "onpromotion": [False, False, False],
        "date": [calendar[0], calendar[1], calendar[4]],
    })

    sales, promo = dense_series(raw, calendar, open_days, store_nbr=1)
    col = sales[100]

    assert col.loc[calendar[0]] == 5.0        # observed
    assert col.loc[calendar[1]] == 0.0        # return clipped to 0
    assert np.isnan(col.loc[calendar[2]])     # closure -> masked
    assert col.loc[calendar[3]] == 0.0        # absent on an OPEN day -> genuine 0 demand
    assert col.loc[calendar[4]] == 7.0        # observed
    assert not promo[100].any()               # nothing on promotion


def test_solve_weights_simplex_and_not_worse_than_uniform() -> None:
    """Step 10 (2): SC weights are non-negative, sum to 1, and beat/tie uniform weights."""
    rng = np.random.default_rng(0)
    n_days, n_donors = 60, 8
    donor = rng.gamma(shape=2.0, scale=5.0, size=(n_days, n_donors))
    # Target is a known positive combination of two donors plus light noise.
    true_w = np.zeros(n_donors)
    true_w[1] = 0.7
    true_w[4] = 0.3
    target = donor @ true_w + rng.normal(0, 0.5, size=n_days)

    weights, method = solve_weights(donor, target, max_iter=500)

    assert weights.shape == (n_donors,)
    assert np.all(weights >= -1e-9)
    assert abs(weights.sum() - 1.0) < 1e-6

    def sse(w: np.ndarray) -> float:
        resid = target - donor @ w
        return float(resid @ resid)

    uniform = np.full(n_donors, 1.0 / n_donors)
    assert sse(weights) <= sse(uniform) + 1e-6
    assert method in {"slsqp", "nnls", "uniform"}


def _make_synth_frames(
    target_pre_pattern: np.ndarray, donor_columns: list[np.ndarray],
    pre: int, promo_len: int, post: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (target, donors) frames for ``fit_event_synth`` from pre-period series.

    The promo/post windows simply continue each series at its pre-period mean so the fit
    (which uses the pre-period only) is exercised without injecting any effect.
    """
    n = pre + promo_len + post
    calendar = pd.date_range("2016-01-01", periods=n, freq="D")
    segment = np.array(["pre"] * pre + ["promo"] * promo_len + ["post"] * post)

    def extend(pre_vals: np.ndarray) -> np.ndarray:
        tail = np.full(promo_len + post, float(np.mean(pre_vals)))
        return np.concatenate([pre_vals, tail])

    target = pd.DataFrame({
        "date": calendar,
        "unit_sales": extend(target_pre_pattern),
        "segment": segment,
    })

    donor_rows = []
    for rank, col_pre in enumerate(donor_columns, start=1):
        donor_rows.append(pd.DataFrame({
            "date": calendar,
            "donor_rank": rank,
            "unit_sales": extend(col_pre),
            "segment": segment,
        }))
    donors = pd.concat(donor_rows, ignore_index=True)
    return target, donors


def test_synth_poor_fit_flag_discriminates() -> None:
    """Step 10 (3): the pre-period SC diagnostic flags a deliberately poor match.

    The target has a strong, slow pre-period oscillation (amplitude ~80% of its mean) that
    survives the 7-day smoothing. When the donors are flat and cannot reproduce it, the
    demeaned synthetic stays near the mean, the normalized pre-period RMSPE is large, and
    ``poor_fit`` is True. When one donor DOES co-move with the target, the fit is near
    perfect and ``poor_fit`` is False -- so the flag genuinely discriminates.
    """
    cfg = load_config()
    cfg = replace(cfg, synth=replace(cfg.synth, bootstrap_draws=20))
    pre, promo_len, post = 90, 10, 10

    t = np.arange(pre, dtype=float)
    target_pre = 100.0 + 80.0 * np.sin(2.0 * np.pi * t / 45.0)  # slow, large swing

    # POOR: five flat donors at different levels -> cannot track the oscillation.
    flat_donors = [np.full(pre, level) for level in (70.0, 85.0, 100.0, 115.0, 130.0)]
    target, donors = _make_synth_frames(target_pre, flat_donors, pre, promo_len, post)
    res_poor, _ = fit_event_synth(target, donors, "POOR", cfg, np.random.default_rng(0))
    assert res_poor["norm_smooth_rmspe"] > cfg.synth.max_pre_rmspe
    assert res_poor["poor_fit"] is True

    # GOOD: one donor reproduces the target's shape (plus flat distractors) -> fit is good.
    good_donors = [target_pre.copy(),
                   np.full(pre, 85.0), np.full(pre, 100.0), np.full(pre, 115.0)]
    target_g, donors_g = _make_synth_frames(target_pre, good_donors, pre, promo_len, post)
    res_good, _ = fit_event_synth(target_g, donors_g, "GOOD", cfg, np.random.default_rng(0))
    assert res_good["norm_smooth_rmspe"] < cfg.synth.max_pre_rmspe
    assert res_good["poor_fit"] is False


class _NoHolidays:
    """Holiday-calendar stub that reports no holidays (keeps the test data-free)."""

    def flags(self, dates: pd.DatetimeIndex, state: str, city: str) -> np.ndarray:
        return np.zeros(len(dates), dtype=bool)


def test_its_counterfactual_matches_hand_calculation() -> None:
    """Step 10 (1): ITS counterfactual matches a HAND-CALCULABLE value.

    On a deterministic pre-period where ``log1p(sales) = a + b*t`` exactly (no noise, no
    closures, no day-of-week or holiday structure), OLS fits with zero residual, so the
    Duan smearing factor is 1 and the back-transformed counterfactual over the promotion
    window equals ``expm1(a + b*t)`` exactly. A KNOWN per-day effect is injected into the
    promo window, so the recovered effect must equal the injected total.
    """
    cfg = load_config()
    cfg = replace(cfg, its=replace(cfg.its, use_oil=False, bootstrap_draws=20))

    pre, promo_len, post = 84, 14, 14
    n = pre + promo_len + post
    calendar = pd.date_range("2016-01-04", periods=n, freq="D")  # a Monday start
    t = np.arange(n, dtype=float)

    a, b = np.log1p(40.0), 0.01
    baseline = np.expm1(a + b * t)          # the true no-promo demand in original units
    injected_per_day = 30.0

    sales = baseline.copy()
    sales[pre:pre + promo_len] += injected_per_day  # known promotion lift

    segment = np.array(["pre"] * pre + ["promo"] * promo_len + ["post"] * post)
    target = pd.DataFrame({"date": calendar, "unit_sales": sales, "segment": segment})
    event = pd.Series({"event_id": "HAND", "state": "S", "city": "C"})

    result, display = fit_event_its(target, event, _NoHolidays(), None, cfg,
                                    np.random.default_rng(1))

    # The promo-window counterfactual equals the hand-computed baseline, day by day.
    promo_rows = display[display["segment"] == "promo"].reset_index(drop=True)
    hand_cf = np.expm1(a + b * t[pre:pre + promo_len])
    assert np.allclose(promo_rows["its_cf"].to_numpy(), hand_cf, rtol=1e-6, atol=1e-4)

    # The recovered effect equals the known injected total (promo_len * per-day lift).
    assert np.isclose(result["promo_effect"], promo_len * injected_per_day,
                      rtol=1e-5, atol=1e-3)
    # No effect was injected into the post window, so its effect is ~0.
    assert abs(result["post_effect"]) < 1e-2
    # A perfect pre-period fit is reflected in the diagnostic.
    assert result["pre_r2"] > 0.999


def test_its_effect_equals_sum_actual_minus_counterfactual() -> None:
    """Step 10 (1, companion): reported effects equal summed actual - counterfactual."""
    cfg = load_config()
    cfg = replace(cfg, its=replace(cfg.its, use_oil=False, bootstrap_draws=50))

    pre, promo_len, post = 70, 10, 10
    n = pre + promo_len + post
    calendar = pd.date_range("2016-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    # A clean trend + weekly pattern pre-period, then an injected promo lift.
    base = 50 + 0.2 * np.arange(n) + 5 * (calendar.dayofweek < 5)
    sales = base + rng.normal(0, 1.0, size=n)
    sales[pre:pre + promo_len] += 40.0  # promotion lift
    sales = np.clip(sales, 0, None)

    segment = np.array(["pre"] * pre + ["promo"] * promo_len + ["post"] * post)
    target = pd.DataFrame({"date": calendar, "unit_sales": sales, "segment": segment})
    event = pd.Series({"event_id": "T1", "state": "SomeState", "city": "SomeCity"})

    result, display = fit_event_its(target, event, _NoHolidays(), None, cfg, rng)

    promo_rows = display[display["segment"] == "promo"]
    post_rows = display[display["segment"] == "post"]
    expected_promo = float((promo_rows["actual"] - promo_rows["its_cf"]).sum())
    expected_post = float((post_rows["actual"] - post_rows["its_cf"]).sum())

    assert np.isclose(result["promo_effect"], expected_promo, atol=1e-6)
    assert np.isclose(result["post_effect"], expected_post, atol=1e-6)
    # The injected lift is large and positive, so the recovered effect must be positive.
    assert result["promo_effect"] > 0
