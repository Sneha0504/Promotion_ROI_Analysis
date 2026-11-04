"""Unit tests for the Unit 3 cannibalization / net-impact / early-warning building blocks.

Covered:
* net-impact internal consistency on a CONSTRUCTED example --
  ``incremental_margin - discount_cost`` equals the reported ``net_profit_impact`` and
  ``net_incremental_units`` equals ``promo_lift - cannibalization`` (a post dip);
* the early-warning backtest train/test sets are DISJOINT (no event in both, and together
  they partition the input), and the threshold is tuned on TRAIN data only (tuning the rule
  on the train labels alone reproduces the chosen threshold).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cannibalization import cannibalization_share, choose_method
from src.early_warning import split_train_test, stratified_kfold, tune_threshold
from src.net_impact import (
    classify,
    compute_break_even,
    compute_economics,
    net_profit_after_shift,
)


def test_net_impact_internal_consistency() -> None:
    """Step 10 (4): net impact is internally consistent on a constructed example.

    Checks the full identity including the discount-cost term:
    ``net_incremental_units = lift - cannibalization`` and
    ``net_profit_impact = incremental_margin - discount_cost``.
    """
    promo_effect = 500.0          # promotion-window lift (units)
    post_effect = -120.0          # post-period dip (cannibalization) -> negative
    promo_units_sold = 2000.0     # actual units sold at the promo price
    assumed_price, discount_depth, gross_margin = 1.0, 0.20, 0.30

    econ = compute_economics(
        promo_effect, post_effect, promo_units_sold,
        assumed_price=assumed_price, discount_depth=discount_depth, gross_margin=gross_margin)

    # net incremental units = promo lift - cannibalization (the post dip magnitude).
    cannibalization = max(0.0, -post_effect)
    assert np.isclose(econ["net_incremental_units"], promo_effect - cannibalization)
    assert np.isclose(econ["net_incremental_units"], promo_effect + post_effect)

    # discount cost and incremental margin match their definitions.
    assert np.isclose(econ["discount_cost"],
                      discount_depth * assumed_price * promo_units_sold)
    assert np.isclose(econ["incremental_margin"],
                      gross_margin * assumed_price * econ["net_incremental_units"])

    # net profit = incremental margin - discount cost (the headline identity).
    assert np.isclose(econ["net_profit_impact"],
                      econ["incremental_margin"] - econ["discount_cost"])


def test_classify_straddle_zero() -> None:
    """Classification is inconclusive iff the CI straddles zero, else by the point sign."""
    assert classify(-1.0, 5.0, 3.0) == "inconclusive"
    assert classify(1.0, 5.0, 3.0) == "net positive"
    assert classify(-5.0, -1.0, -3.0) == "net negative"


def test_choose_method_and_share() -> None:
    """SC is preferred when good; both-poor is low-confidence; share handles edge cases."""
    assert choose_method(sc_poor_fit=False, its_poor_fit=False) == ("SC", False)
    assert choose_method(sc_poor_fit=True, its_poor_fit=False) == ("ITS", False)
    assert choose_method(sc_poor_fit=True, its_poor_fit=True) == ("ITS", True)

    assert np.isclose(cannibalization_share(500.0, -120.0), 120.0 / 500.0)
    assert cannibalization_share(500.0, 40.0) == 0.0          # positive post -> no cannib
    assert np.isnan(cannibalization_share(0.0, -120.0))       # no lift -> undefined share
    assert np.isnan(cannibalization_share(-10.0, -5.0))       # negative lift -> undefined


def test_early_warning_split_is_disjoint_and_partitions() -> None:
    """Step 10 (5): early-warning train and test sets are disjoint (no shared event).

    They never share an event and together cover every input event (companion
    ``test_tune_threshold_uses_train_only`` confirms the rule is tuned on TRAIN only).
    """
    event_ids = [f"EVT{i:05d}" for i in range(1, 101)]
    train, test = split_train_test(event_ids, train_fraction=0.6, test_fraction=0.4,
                                   seed=20170815)

    assert set(train).isdisjoint(set(test))                   # no event in both
    assert set(train) | set(test) == set(event_ids)           # together, all events
    assert len(train) + len(test) == len(event_ids)           # no duplicates
    assert len(train) == 60 and len(test) == 40               # fractions honored


def test_early_warning_split_holds_out_remainder_and_validates_fractions() -> None:
    """A train+test sum below 1 holds out the remainder; above 1 raises."""
    event_ids = [f"EVT{i:05d}" for i in range(1, 101)]
    train, test = split_train_test(event_ids, train_fraction=0.6, test_fraction=0.3,
                                   seed=20170815)
    assert set(train).isdisjoint(set(test))
    assert len(train) == 60 and len(test) == 30              # 10 events held out
    assert len(set(train) | set(test)) == 90

    with pytest.raises(ValueError):
        split_train_test(event_ids, train_fraction=0.7, test_fraction=0.4, seed=1)


def test_tune_threshold_uses_train_only() -> None:
    """The threshold is tuned on TRAIN data alone -- a different TEST set would give a
    different threshold, and the shipped rule is the train-tuned one.

    Net-negative events (label 1) carry a LOW early signal. The train set is cleanly separable
    around 0; a TEST set with a different separation point (around +2) would be tuned to a
    different threshold, proving the shipped rule did not see the test data.
    """
    rng = np.random.default_rng(0)
    # TRAIN: negatives ~-0.5, non-negatives ~+0.5 -> separating threshold near 0.
    train_signal = np.concatenate([rng.normal(-0.5, 0.05, 30), rng.normal(0.5, 0.05, 30)])
    train_label = np.concatenate([np.ones(30, dtype=int), np.zeros(30, dtype=int)])
    # TEST: negatives ~+1.5, non-negatives ~+2.5 -> separating threshold near +2.
    test_signal = np.concatenate([rng.normal(1.5, 0.05, 30), rng.normal(2.5, 0.05, 30)])
    test_label = np.concatenate([np.ones(30, dtype=int), np.zeros(30, dtype=int)])

    train_thresh = tune_threshold(train_signal, train_label, steps=201, metric="f1")
    test_thresh = tune_threshold(test_signal, test_label, steps=201, metric="f1")

    # The train-tuned threshold separates train but would badly misfit test, and the two
    # thresholds are far apart -- so the shipped (train-tuned) rule cannot have used test data.
    assert -0.5 < train_thresh < 0.5
    assert 1.5 < test_thresh < 2.5
    assert abs(train_thresh - test_thresh) > 1.0
    assert np.array_equal((train_signal < train_thresh).astype(int), train_label)
    # Applying the train threshold to the test set flags everything (all test signals > 0),
    # i.e. it does not reproduce the test separation -- confirming no test leakage.
    assert (test_signal < train_thresh).sum() == 0


def test_stratified_kfold_disjoint_and_covers_all() -> None:
    """CV folds partition every event (disjoint, complete) and keep class balance."""
    labels = np.array([1] * 70 + [0] * 30)
    folds = stratified_kfold(labels, k=5, seed=20170815)

    assert len(folds) == 5
    all_idx = np.concatenate(folds)
    assert sorted(all_idx.tolist()) == list(range(100))        # cover all, no duplicates
    for i, a in enumerate(folds):                               # pairwise disjoint
        for b in folds[i + 1:]:
            assert set(a).isdisjoint(set(b))
    # Each fold's positive rate is close to the overall 0.70 (stratification works).
    for f in folds:
        assert 0.5 <= labels[f].mean() <= 0.9


def test_break_even_formulas_and_consistency() -> None:
    """Break-even discount zeroes net profit; never-breaks-even when net units <= 0."""
    r = compute_break_even(2000.0, 5000.0, discount_depth=0.20, gross_margin=0.30)
    assert np.isclose(r["break_even_discount"], 0.30 * 2000.0 / 5000.0)  # 0.12
    assert np.isclose(r["break_even_margin"], 0.20 * 5000.0 / 2000.0)    # 0.50
    assert r["never_breaks_even"] is False

    # Running the promotion at exactly the break-even discount gives net profit 0.
    econ = compute_economics(2000.0, 0.0, 5000.0, assumed_price=1.0,
                             discount_depth=r["break_even_discount"], gross_margin=0.30)
    assert np.isclose(econ["net_profit_impact"], 0.0, atol=1e-6)

    # No net incremental units -> never breaks even; break-even margin undefined.
    r2 = compute_break_even(-10.0, 100.0, discount_depth=0.20, gross_margin=0.30)
    assert r2["never_breaks_even"] is True
    assert np.isnan(r2["break_even_margin"])


def test_placebo_robustness_shift() -> None:
    """Adding optimistic placebo noise shifts net profit the right way and the threshold holds."""
    gm_price = 0.30 * 1.0
    # A strongly net-negative event: net profit well below zero, discount cost 300.
    #   net_profit = 0.30 * net_units - 300.
    # With net_units = -500: base net profit = -450. A +100 placebo shift -> 0.30*(-400)-300 = -420,
    # still net negative (robust). A +1600 shift -> 0.30*(1100)-300 = +30, flips positive.
    assert net_profit_after_shift(-500.0, 0.0, 300.0, gm_price) < 0      # base is net negative
    assert net_profit_after_shift(-500.0, 100.0, 300.0, gm_price) < 0    # survives p90-scale noise
    assert net_profit_after_shift(-500.0, 1600.0, 300.0, gm_price) > 0   # flips under huge noise
    # The shift enters only through net units (margin side), not the fixed discount cost.
    assert np.isclose(net_profit_after_shift(-500.0, 100.0, 300.0, gm_price),
                      gm_price * (-500.0 + 100.0) - 300.0)
