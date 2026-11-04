"""Unit 3, step 3: early-warning rule + backtest (PLAN sec. 6e, Step 9).

A deliberately simple, explainable rule that looks only at the **first ``early_days`` days**
of a promotion and flags promotions likely to finish **net negative** -- so a buyer could pull
or adjust a losing promotion before it ends, instead of finding out afterward.

Early signal (documented; uses NO data from day ``early_days + 1`` onward)
------------------------------------------------------------------------
For each event, the daily counterfactual of the **chosen method** (SC where its pre-fit is
good, else ITS -- the same choice recorded in ``cannibalization.parquet``) gives a daily lift
``actual - counterfactual``. The signal is the **early lift per day, normalized by the
pre-period baseline level**::

    early_signal = (sum of (actual - cf) over the first early_days promo days / early_days)
                   / pre_period_mean_daily_sales

Normalizing by the pre-period mean makes the signal comparable across high- and low-volume
item-stores (a raw unit lift is not). A low or negative early signal means the promotion is
barely lifting sales relative to its own baseline -- the early warning sign of a net loser.

Target label (documented)
-------------------------
The label is the full-window net-impact classification: ``1`` = **net negative**, ``0`` =
not net negative. **Inconclusive promotions are GROUPED with the negatives-excluded class
(label 0)**: the rule's job is to catch *confidently* net-negative promotions, and an
inconclusive one is not confidently a loser. The inconclusive count is reported so the
grouping is transparent.

Backtest (documented, honest)
-----------------------------
Evaluation is **stratified K-fold cross-validation** (``early_warning.k_folds``, stratified on the
net-negative label, seeded by ``random_seed``). For each fold the threshold is tuned on the other
folds and evaluated on the held-out fold, so every event gets exactly one out-of-fold prediction.
The CV is run under **both** tuning criteria -- **F1** of the net-negative class and **Youden's J**
(sensitivity + specificity - 1) -- and both are reported. The **deployable** metric is chosen by
``early_warning.tune_metric``: ``"auto"`` (default) prefers whichever gives a real pooled-precision
lift over the base rate at usable recall (F1 tends to flag almost everything on this imbalanced
sample, so Youden is usually more discriminating); or it can be forced to ``"f1"``/``"youden"``. If
neither clears the lift threshold, that is reported plainly (the rule adds little over the base
rate). Precision/recall/F1 are reported as **mean +/- std across folds** plus the **pooled
out-of-fold** confusion matrix; a single deployable threshold (tuned on all scored events with the
chosen metric) is reported with the CV as its out-of-sample estimate. Folds are disjoint and cover
every scored event (enforced and unit-tested). This reduces -- but does not eliminate -- the
small/imbalanced-sample fragility, and the base-rate caveat is kept.
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

# Positive class = "net negative" (the thing the early warning tries to catch).
NET_NEGATIVE = "net negative"


def split_train_test(
    event_ids: list[str], train_fraction: float, test_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split event ids into disjoint train / test lists, seeded for reproducibility.

    The two fractions must sum to at most 1.0. ``train_fraction`` of the events are assigned to
    train and ``test_fraction`` to test; any remainder is held out of the backtest entirely.
    With the default 0.6 / 0.4 the two sets partition the full input.

    Args:
        event_ids: all event ids to split.
        train_fraction: share assigned to the train set.
        test_fraction: share assigned to the (disjoint) test set.
        seed: RNG seed so the split is reproducible.

    Returns:
        ``(train_ids, test_ids)`` -- always disjoint.

    Raises:
        ValueError: if the fractions are negative or sum to more than 1.0.
    """
    if train_fraction < 0 or test_fraction < 0:
        raise ValueError("train_fraction and test_fraction must be non-negative.")
    if train_fraction + test_fraction > 1.0 + 1e-9:
        raise ValueError(
            f"train_fraction + test_fraction must be <= 1.0 "
            f"(got {train_fraction} + {test_fraction}).")
    rng = np.random.default_rng(seed)
    shuffled = list(event_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_fraction))
    n_test = min(int(round(n * test_fraction)), n - n_train)
    return shuffled[:n_train], shuffled[n_train:n_train + n_test]


def stratified_kfold(labels: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Return k disjoint test-index arrays, stratified on ``labels`` and seeded.

    Within each class, shuffled indices are round-robin assigned to folds, so every fold has a
    near-identical class balance. The returned arrays are disjoint and together cover every index.

    Args:
        labels: per-sample integer labels to stratify on.
        k: number of folds (>= 2).
        seed: RNG seed for reproducibility.

    Returns:
        A list of ``k`` sorted index arrays (a partition of ``range(len(labels))``).
    """
    if k < 2:
        raise ValueError(f"k_folds must be >= 2 (got {k}).")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in np.unique(labels):
        cls_idx = idx[labels == cls]
        rng.shuffle(cls_idx)
        for i, ix in enumerate(cls_idx):
            folds[i % k].append(int(ix))
    return [np.array(sorted(f), dtype=int) for f in folds]


def _f1(conf: dict[str, int]) -> float:
    """F1 of the net-negative class from confusion counts (NaN if undefined)."""
    precision, recall = _precision_recall(conf)
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


def compute_early_signal(
    display_series: pd.DataFrame, cannib: pd.DataFrame, early_days: int
) -> pd.DataFrame:
    """Compute the per-event early signal from the first ``early_days`` promotion days.

    Uses the chosen method's daily counterfactual and normalizes the early lift-per-day by the
    event's pre-period mean daily sales. Returns a frame of
    ``event_id, early_signal, early_cum_lift, pre_baseline``.
    """
    method_by_event = dict(zip(cannib["event_id"], cannib["chosen_method"], strict=True))
    ds = display_series.sort_values(["event_id", "date"])

    rows: list[dict[str, object]] = []
    for event_id, grp in ds.groupby("event_id", sort=False):
        method = method_by_event.get(event_id, "SC")
        cf_col = "sc_cf" if method == "SC" else "its_cf"

        pre = grp[grp["segment"] == "pre"]
        pre_baseline = float(pre["actual"].mean(skipna=True))

        promo = grp[grp["in_promo"]].head(early_days)
        lift = (promo["actual"] - promo[cf_col]).dropna()
        n_days = int(len(lift))
        early_cum_lift = float(lift.sum()) if n_days else float("nan")
        per_day = early_cum_lift / n_days if n_days else float("nan")
        early_signal = (per_day / pre_baseline
                        if pre_baseline and np.isfinite(pre_baseline) and pre_baseline > 0
                        else float("nan"))

        rows.append({
            "event_id": event_id,
            "early_signal": early_signal,
            "early_cum_lift": early_cum_lift,
            "pre_baseline": pre_baseline,
        })
    return pd.DataFrame(rows)


def _confusion(actual: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    """Confusion counts with the net-negative class as positive."""
    tp = int(np.sum((predicted == 1) & (actual == 1)))
    fp = int(np.sum((predicted == 1) & (actual == 0)))
    fn = int(np.sum((predicted == 0) & (actual == 1)))
    tn = int(np.sum((predicted == 0) & (actual == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _precision_recall(conf: dict[str, int]) -> tuple[float, float]:
    """Precision and recall for the net-negative class from confusion counts."""
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall


def _score(actual: np.ndarray, predicted: np.ndarray, metric: str) -> float:
    """Tuning criterion on a split: F1 of the net-negative class, or Youden's J."""
    conf = _confusion(actual, predicted)
    precision, recall = _precision_recall(conf)
    if metric == "youden":
        tn, fp = conf["tn"], conf["fp"]
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        j = (0.0 if np.isnan(recall) else recall) + (0.0 if np.isnan(specificity)
                                                      else specificity) - 1.0
        return j
    # default: F1
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def tune_threshold(
    train_signal: np.ndarray, train_label: np.ndarray, steps: int, metric: str
) -> float:
    """Pick the threshold maximizing ``metric`` on TRAIN; flag when ``signal < threshold``.

    Candidate thresholds are scanned across the observed train-signal range. Ties are broken
    toward the lower threshold (fewer false positives). Uses only the training data.
    """
    finite = train_signal[np.isfinite(train_signal)]
    if finite.size == 0:
        return float("nan")
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    candidates = np.linspace(lo, hi, steps)

    best_thresh, best_score = candidates[0], -np.inf
    for thresh in candidates:
        predicted = (train_signal < thresh).astype(int)
        score = _score(train_label, predicted, metric)
        if score > best_score:
            best_score, best_thresh = score, float(thresh)
    return best_thresh


def _failure_cases(frame: pd.DataFrame, kind: str, limit: int = 5) -> list[dict[str, object]]:
    """Characterize false positives / false negatives on the test set (brief)."""
    if kind == "false_positive":
        sub = frame[(frame["predicted_flag"] == 1) & (frame["actual_label"] == 0)]
    else:  # false_negative
        sub = frame[(frame["predicted_flag"] == 0) & (frame["actual_label"] == 1)]
    cols = ["event_id", "early_signal", "net_profit_impact", "classification"]
    cols = [c for c in cols if c in sub.columns]
    out: list[dict[str, object]] = []
    for row in sub[cols].head(limit).itertuples(index=False):
        out.append({c: (round(float(v), 3) if isinstance(v, (int, float, np.floating))
                        else v) for c, v in zip(cols, row, strict=True)})
    return out


def _metrics_block(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    """Standard metrics block (n, class balance, confusion, precision, recall) for a set."""
    conf = _confusion(actual, predicted)
    precision, recall = _precision_recall(conf)
    n = int(len(actual))
    n_pos = int(actual.sum())
    return {
        "n": n,
        "n_net_negative": n_pos,
        "n_not_net_negative": n - n_pos,
        "base_rate_net_negative": round(n_pos / n, 3) if n else float("nan"),
        "confusion_matrix": conf,
        "precision": None if np.isnan(precision) else round(precision, 3),
        "recall": None if np.isnan(recall) else round(recall, 3),
    }


def _run_cv_for_metric(
    fold_test_idx: list[np.ndarray], labels: np.ndarray, sig: np.ndarray,
    steps: int, metric: str,
) -> dict[str, object]:
    """Run the stratified K-fold CV under one tuning ``metric`` ("f1" or "youden").

    Returns the per-fold metrics, mean +/- std, pooled out-of-fold block, fold thresholds, the
    deployable threshold (tuned on all scored events), and the out-of-fold prediction array
    (``oof_pred``, stripped before JSON serialization).
    """
    oof = np.zeros(len(labels), dtype=int)
    per_fold: list[dict[str, object]] = []
    thresholds: list[float] = []
    for f, test_idx in enumerate(fold_test_idx):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[test_idx] = False
        thr = tune_threshold(sig[train_mask], labels[train_mask], steps, metric)
        thresholds.append(thr)
        pred = (sig[test_idx] < thr).astype(int)
        oof[test_idx] = pred
        conf = _confusion(labels[test_idx], pred)
        precision, recall = _precision_recall(conf)
        per_fold.append({
            "fold": f,
            "n": int(len(test_idx)),
            "threshold": None if np.isnan(thr) else round(thr, 4),
            "precision": None if np.isnan(precision) else round(precision, 3),
            "recall": None if np.isnan(recall) else round(recall, 3),
            "f1": None if np.isnan(_f1(conf)) else round(_f1(conf), 3),
            "base_rate_net_negative": round(float(labels[test_idx].mean()), 3),
            "confusion_matrix": conf,
        })

    def ms(key: str) -> tuple[float, float]:
        vals = [m[key] for m in per_fold if m[key] is not None]
        return (round(float(np.mean(vals)), 3), round(float(np.std(vals)), 3)) if vals else (
            float("nan"), float("nan"))

    prec_mean, prec_std = ms("precision")
    rec_mean, rec_std = ms("recall")
    f1_mean, f1_std = ms("f1")
    deployable = tune_threshold(sig, labels, steps, metric)
    return {
        "metric": metric,
        "k": len(fold_test_idx),
        "fold_thresholds": [None if np.isnan(t) else round(t, 4) for t in thresholds],
        "median_fold_threshold": round(float(np.nanmedian(thresholds)), 4),
        "per_fold": per_fold,
        "precision_mean": prec_mean, "precision_std": prec_std,
        "recall_mean": rec_mean, "recall_std": rec_std,
        "f1_mean": f1_mean, "f1_std": f1_std,
        "pooled_out_of_fold": _metrics_block(labels, oof),
        "deployable_threshold": None if np.isnan(deployable) else round(deployable, 4),
        "oof_pred": oof,
    }


def build_early_warning(cfg: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the per-event early-warning table and the stratified K-fold CV summary."""
    out = cfg.paths.data_processed
    display_series = pd.read_parquet(out / "display_series.parquet")
    cannib = pd.read_parquet(out / "cannibalization.parquet")
    net_impact = pd.read_parquet(out / "net_impact.parquet")

    ew = cfg.early_warning
    signals = compute_early_signal(display_series, cannib, ew.early_days)
    frame = signals.merge(
        net_impact[["event_id", "classification", "net_profit_impact"]],
        on="event_id", how="inner")
    frame["actual_label"] = (frame["classification"] == NET_NEGATIVE).astype(int)

    # Only events with a usable early signal can be scored by the rule.
    usable = frame[frame["early_signal"].notna()].copy().reset_index(drop=True)
    labels = usable["actual_label"].to_numpy(dtype=int)
    sig = usable["early_signal"].to_numpy(dtype=float)
    k = ew.k_folds

    # Stratified K-fold cross-validation (the fold split is independent of the tuning metric).
    # Run the CV under BOTH F1 and Youden's J so both can be reported; pick the deployable metric.
    fold_test_idx = stratified_kfold(labels, k, cfg.random_seed)
    cv_by_metric = {m: _run_cv_for_metric(fold_test_idx, labels, sig, ew.threshold_grid_steps, m)
                    for m in ("f1", "youden")}
    base = cv_by_metric["f1"]["pooled_out_of_fold"]["base_rate_net_negative"]

    def lift_recall(cv: dict[str, object]) -> tuple[float, float]:
        pooled_block = cv["pooled_out_of_fold"]
        precision = pooled_block["precision"] if pooled_block["precision"] is not None else 0.0
        recall = pooled_block["recall"] if pooled_block["recall"] is not None else 0.0
        return precision - base, recall

    # Deployable metric: forced by config, or (default "auto") the one with a real precision lift
    # over the base rate at usable recall -- preferring Youden on a tie -- else neither is useful.
    if ew.tune_metric in ("f1", "youden"):
        deployable_metric = ew.tune_metric
        d_lift, d_recall = lift_recall(cv_by_metric[deployable_metric])
        beats_base_rate = bool(d_recall >= ew.recall_floor and d_lift >= ew.min_precision_lift)
        selection_reason = f"forced by config tune_metric='{ew.tune_metric}'."
    else:
        eligible = []
        for m in ("youden", "f1"):
            m_lift, m_recall = lift_recall(cv_by_metric[m])
            if m_recall >= ew.recall_floor and m_lift >= ew.min_precision_lift:
                eligible.append((round(m_lift, 4), m))
        if eligible:
            eligible.sort(reverse=True)
            deployable_metric = eligible[0][1]
            d_lift, d_recall = lift_recall(cv_by_metric[deployable_metric])
            beats_base_rate = True
            selection_reason = (
                f"auto: {deployable_metric} gives the largest pooled precision lift over the base "
                f"rate (+{round(d_lift, 3)}) at usable recall ({round(d_recall, 3)} >= "
                f"{ew.recall_floor}).")
        else:
            deployable_metric = "youden"
            d_lift, d_recall = lift_recall(cv_by_metric["youden"])
            beats_base_rate = False
            selection_reason = (
                f"auto: neither F1 nor Youden clears a +{ew.min_precision_lift} precision lift over "
                f"the base rate at recall >= {ew.recall_floor}; the early rule adds little over "
                f"flagging everything. Deployable set to Youden (the more discriminating "
                f"threshold), reported honestly as near-trivial on this sample.")

    chosen = cv_by_metric[deployable_metric]
    oof_pred = chosen["oof_pred"]
    deployable_threshold = chosen["deployable_threshold"]

    # Fold ids + chosen-metric OOF predictions back onto the full frame (unscored keep fold -1).
    usable["fold"] = -1
    for f, test_idx in enumerate(fold_test_idx):
        usable.loc[test_idx, "fold"] = f
    oof_map = dict(zip(usable["event_id"], oof_pred, strict=True))
    fold_map = dict(zip(usable["event_id"], usable["fold"], strict=True))
    frame["fold"] = frame["event_id"].map(fold_map).fillna(-1).astype(int)
    frame["predicted_flag"] = frame["event_id"].map(oof_map).fillna(0).astype(int)

    pooled = chosen["pooled_out_of_fold"]
    in_sample_thr = deployable_threshold if deployable_threshold is not None else np.inf
    in_sample = _metrics_block(labels, (sig < in_sample_thr).astype(int))

    pooled_prec = pooled["precision"] if pooled["precision"] is not None else 0.0
    verdict = (
        f"Under {k}-fold CV the deployable rule ({deployable_metric}, threshold "
        f"{deployable_threshold}) gives pooled out-of-fold precision {pooled['precision']} / "
        f"recall {pooled['recall']} vs a {base} base rate -- "
        + ("a real precision lift over flagging everything."
           if beats_base_rate else
           "essentially no precision lift; the rule adds little over flagging everything."))
    base_rate_comparison = (
        f"Pooled out-of-fold precision {pooled['precision']} vs the {base} base rate "
        f"(+{round(pooled_prec - base, 3)}); flagging EVERY promotion already yields precision "
        f"{base} at recall 1.0. {verdict}")

    disjoint = all(set(a).isdisjoint(set(b)) for i, a in enumerate(fold_test_idx)
                   for b in fold_test_idx[i + 1:])
    covers_all = (sorted(np.concatenate(fold_test_idx).tolist()) == list(range(len(usable))))

    def strip_oof(cv: dict[str, object]) -> dict[str, object]:
        return {key: val for key, val in cv.items() if key != "oof_pred"}

    n_inconclusive = int((frame["classification"] == "inconclusive").sum())
    scored_frame = frame[frame["fold"] >= 0]
    summary: dict[str, object] = {
        "rule": ("Flag a promotion as likely net-negative when its normalized early "
                 f"lift-per-day over the first {ew.early_days} promo days is below the tuned "
                 "threshold (early_signal < threshold)."),
        "signal_definition": ("(mean daily (actual - chosen-method counterfactual) over the "
                              "first early_days promo days) / pre-period mean daily sales."),
        "label_definition": ("1 = net negative; 0 = net positive OR inconclusive "
                             "(inconclusive grouped with the non-negative class)."),
        "early_days": ew.early_days,
        "tune_metric_config": ew.tune_metric,
        "tune_metric": deployable_metric,  # the metric actually deployed (alias for consumers)
        "deployable_metric": deployable_metric,
        "deployable_metric_reason": selection_reason,
        "beats_base_rate": beats_base_rate,
        "verdict": verdict,
        "threshold_grid_steps": ew.threshold_grid_steps,
        "k_folds": k,
        "deployable_threshold": deployable_threshold,
        "deployable_threshold_note": ("One deployable threshold, tuned on ALL scored events with "
                                      "the chosen metric; the CV estimates its out-of-sample "
                                      "performance. Both metrics' CV are under cv_by_metric."),
        "threshold": deployable_threshold,
        "n_total_events": int(len(frame)),
        "n_scored": int(len(usable)),
        "n_unscored_no_signal": int((frame["fold"] < 0).sum()),
        "n_inconclusive_grouped_as_label0": n_inconclusive,
        "cv": strip_oof(chosen),
        "cv_by_metric": {m: strip_oof(cv_by_metric[m]) for m in ("f1", "youden")},
        # `test` = pooled out-of-fold under the deployable metric; `train` = in-sample at the
        # deployable threshold (optimistic reference). Kept as named blocks for downstream use.
        "test": pooled,
        "train": in_sample,
        "base_rate_comparison": base_rate_comparison,
        "disjoint_folds": bool(disjoint),
        "folds_cover_all_scored": bool(covers_all),
        "failure_cases_test": {
            "false_positives": _failure_cases(scored_frame, "false_positive"),
            "false_negatives": _failure_cases(scored_frame, "false_negative"),
        },
        "caveats": ("Small, imbalanced sample; stratified K-fold CV reduces but does not eliminate "
                    "the small-sample fragility (each held-out fold has ~" + str(len(usable) // k) +
                    " events). Thresholds are tuned only on training folds; the labels rest on "
                    "PERCENTILE bootstrap CIs."),
    }

    cols = ["event_id", "fold", "early_signal", "early_cum_lift", "pre_baseline",
            "predicted_flag", "actual_label", "classification", "net_profit_impact"]
    return frame[cols], summary


def run_early_warning(cfg: Config) -> pd.DataFrame:
    """Compute the early-warning backtest, write the per-event table and summary JSON."""
    frame, summary = build_early_warning(cfg)
    out = cfg.paths.data_processed
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "early_warning.parquet", index=False)
    with (out / "early_warning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    for m in ("f1", "youden"):
        cvm = summary["cv_by_metric"][m]
        pool = cvm["pooled_out_of_fold"]
        logger.info("EW CV [%s]: threshold=%s | precision=%.3f+/-%.3f recall=%.3f+/-%.3f "
                    "f1=%.3f+/-%.3f | pooled precision=%s recall=%s confusion=%s",
                    m, cvm["deployable_threshold"], cvm["precision_mean"], cvm["precision_std"],
                    cvm["recall_mean"], cvm["recall_std"], cvm["f1_mean"], cvm["f1_std"],
                    pool["precision"], pool["recall"], pool["confusion_matrix"])
    logger.info("EW deployable=%s (threshold=%s) | beats base rate=%s | %s",
                summary["deployable_metric"], summary["deployable_threshold"],
                summary["beats_base_rate"], summary["deployable_metric_reason"])
    logger.info("EW verdict: %s", summary["verdict"])
    logger.info("Folds disjoint=%s | cover all scored=%s | scored n=%d (base rate %.3f)",
                summary["disjoint_folds"], summary["folds_cover_all_scored"],
                summary["n_scored"], summary["test"]["base_rate_net_negative"])
    return frame


def main() -> None:
    """CLI entry point for the ``warning`` Makefile target."""
    parser = argparse.ArgumentParser(description="Early-warning rule + backtest.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    run_early_warning(cfg)


if __name__ == "__main__":
    main()
