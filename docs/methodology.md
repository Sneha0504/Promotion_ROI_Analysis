# Methodology — Promotion ROI Analysis

Internal methodology document (not linked from the public README). Every number below is
reproduced from the committed result tables and summary files in `data/processed/`, produced by
an end-to-end run (`make all`). Run date: 2026-11-04.

Two conventions hold throughout:

- **Economics are modeled assumptions.** The Favorita dataset has **no price and no margin**
  columns, so the discount cost and profit figures are computed from clearly-labeled,
  configurable assumptions (price normalized to 1.0 → results in **margin-units**; discount depth
  20%; gross margin 30%). They are surfaced as assumptions on the dashboard and stress-tested with
  a sensitivity grid.
- **All effect confidence intervals are percentile bootstrap intervals**, labeled as such
  wherever they appear.

---

## 1. Data and sample selection (`src/load.py`)

Source: Corporación Favorita Grocery Sales Forecasting (Kaggle) — ~125M daily `(store, item,
date)` unit-sales rows with a per-row `onpromotion` flag, plus store metadata and a
holidays/events calendar. The large CSVs are scanned with DuckDB (chunked, bounded memory/temp)
and never loaded whole.

**Profiling facts** (`selection_stats.json`): 125,497,040 rows spanning 2013-01-01 → 2017-08-15;
54 stores; 4,036 items; 0 null and 0 zero sales rows (Favorita stores no zero rows — absence is
zero demand); 7,795 negative-sales rows (returns); the `onpromotion` flag is null before
2014-04-01, so the analysis window starts there (`reliable_flag_start`).

**Promotion event** = a maximal run of consecutive `onpromotion = True` days for one
`(item, store)`, merging gaps of ≤ 1 day (`promo_flag_gap_tolerance`). Selection funnel
(`selection_stats.json`): 3,455,720 candidate runs → 427,674 of length ≥ `min_promo_days` (3) and
≤ `max_promo_days` (60) → 405,849 with ≥ 90 pre-days of history passing the series-level
non-zero-fraction guard (`min_pre_nonzero_frac` = 0.5 computed over *stored* rows — a near no-op
proxy, since Favorita stores no zero rows; genuine zero-demand density is handled later by the
dense-calendar reindex, see `data_quality_report.md` rule 3) → 46,728 with a **clean window** (no other promotion inside the pre/promo/post window) → 42,807 with
the store open across the window → capped at **300** analyzable events (`max_events`).

The 300 events span **34 stores, 81 items, 7 product families** (Beverages, Bread/Bakery,
Cleaning, Frozen Foods, Grocery I, Liquor/Wine/Beer, Personal Care), with promotions from
2014-08 to 2017-07.

**Carried-forward data-quality rules:**
1. **Sparse storage → calendar reindex.** Each target and donor series is reindexed to the full
   daily calendar; absent days on trading days are filled with **0 unit_sales**.
2. **Store-closure masking.** A `(store, date)` with no row in `transactions.csv` is a closure;
   all of that store's series are set to **NaN** (never 0) and masked days are excluded from donor
   selection, model fitting and every effect sum (12,493 closure day-series masked).
3. **Returns.** Negative `unit_sales` are clipped to 0 for demand modeling.

---

## 2. Analysis panel and donor pools (`src/build_panel.py`)

For each event a daily panel is built over its full `[pre_start, post_end]` window (90 pre-days +
promotion length + 28 post-days = 121–174 days) containing the promoted target plus a pool of
donor item-stores (`donors` config):

- Same product **family** as the target (primary axis; `same_class` optional, off).
- **Same store first**, expanded to same-**cluster** stores only when the same-store pool is below
  `min_donors` (5). 297/300 events were served entirely from the same store; 3 needed expansion.
- A donor must **never be on promotion** anywhere in the window, must sell on ≥ 30% of its
  observed pre-period days, and must be non-degenerate.
- Donors are ranked by **pre-period-only** correlation with the target and capped at `max_donors`
  (30). **0 of 300** events were dropped for lack of a donor pool; median 30 donors per event.

---

## 3. Interrupted time series (`src/interrupted_ts.py`)

- **Model.** OLS on the **pre-period only** of `log1p(unit_sales)` ~ linear time trend +
  day-of-week dummies (Monday baseline) + a store-specific **holiday** indicator (national always;
  regional/local matched on state/city; `transferred` holidays dropped, `Work Day` excluded) + an
  **optional oil** regressor (default off). Masked closure days are dropped from the fit.
- **Back-transformation.** The central counterfactual uses **Duan's smearing estimator** to
  correct `log1p` retransformation bias: for `y = log1p(sales)`, mean sales =
  `expm1(linpred)·S + (S − 1)` with `S = mean(exp(residual))` — the mean, not the median, clipped
  at 0. (The earlier `expm1(linpred)·S` form is wrong because it equals `exp(linpred)·S − S`.)
- **Effect.** `actual − counterfactual`, summed over the promotion window and, separately, the
  post-period, in original units.
- **Confidence interval** (95%): a **moving-block residual bootstrap** (1000 draws, 7-day blocks)
  that re-fits on resampled pre-period residuals (parameter uncertainty) and adds block-resampled
  residual noise over the horizon (predictive uncertainty + weekly autocorrelation). Bootstrap
  draws already carry an empirical residual, so they are back-transformed with **plain `expm1`**
  (no smearing — averaging `exp(residual)` over draws already reproduces the smearing factor;
  applying it again would double-count the retransformation and bias the interval low).
- **Plausibility guard (two criteria).** A `log1p`-linear trend back-transforms to an
  **exponential**, which can compound to absurd levels projected out-of-sample. A projected daily
  counterfactual is flagged **`cf_implausible`** (which also sets `poor_fit`) when it exceeds
  **either** `cf_cap_multiple` (10) × the pre-period **max** daily sales **or**
  `cf_cap_mean_multiple` (6) × the pre-period **mean** daily sales; the counterfactual is then
  **capped at the binding (smaller) ceiling** — for both the point estimate **and every bootstrap
  draw**, so an exploding draw cannot inflate the CI beyond the capped point. The point-in-interval
  invariant (point ∈ [lo, hi]) is enforced by clamping each bound to include the point (for a few
  capped events a bound equals the point exactly — a consistency guard, not a genuine endpoint).
  The max-based criterion catches blow-ups above the historical peak; the mean-based criterion
  catches milder artifacts that stay under the max cap but still balloon far above the baseline
  level (e.g. EVT00144, whose counterfactual reached ~8× the pre-period peak and ~25× the mean).
  **The 6× mean multiple was chosen post-hoc on this run** (good-fit events top out at ~5.3× the
  mean and stay within the pre-period range, so 6× catches the genuine artifacts — including
  EVT00144 — without demoting any good-fit event); it should be re-checked on new data. On this run
  **15/300** events hit the guard (14 were already poor-fit; EVT00144 is the only newly-caught
  good-fit event).
- **Diagnostics.** Pre-period R², RMSE, normalized RMSE, Durbin–Watson, Ljung–Box (14 lags).
  `poor_fit` when R² < 0.20 or normalized RMSE > 1.50 **or** `cf_implausible`; `autocorr_flag`
  when Ljung–Box p < 0.05 or DW ∉ (1, 3). OLS is the primary method; SARIMA is the documented
  fallback for severely autocorrelated series (surfaced via `autocorr_flag`, not silently refit).

Results (`its_results.parquet`): median pre-period R² **0.373**; **90/300** flagged poor-fit
(**15** of them for implausible counterfactuals); 144/300 flagged autocorrelated; median
promotion-window effect +228 units.

---

## 4. Synthetic control (`src/synthetic_control.py`)

- **Weights.** Non-negative, **summing to 1**, solved by SciPy **SLSQP** over the simplex. The
  raw objective on grocery sales is O(1e4–1e6), on which SLSQP stalls, so the problem is **scaled**
  (divided by `max|target|`) to O(1). The returned weights are the **best of several candidates by
  the actual objective** (SLSQP, normalized-NNLS, uniform), so the fit is **never worse than
  uniform**. On this run all 300 events solved by SLSQP; weights are sparse (median 2.2 effective
  donors).
- **Fitting on the de-noised signal.** Daily item sales are Poisson-like and too noisy to match
  day-by-day, so weights are fit on a **7-day rolling mean of the pre-period only** (no promo
  leakage); the control tracks the trend and weekly pattern. **Effects are still computed on raw
  daily sales.**
- **Pre-period level fixed effect** (`allow_intercept`, default on). The targets are high-volume
  items whose level exceeds the convex hull of their (mostly smaller) same-family donors, so a
  pure sum-to-1 combination cannot reach the level. Series are demeaned before the weight fit and
  the target's pre-mean is added back: `synth(t) = μ_target + (donor(t) − μ_donor)·w`. Weights
  stay on the simplex; the synthetic's pre-mean equals the target's by construction. (With
  `allow_intercept: false` the module is textbook sum-to-1 SC.)
- **Credibility check.** Pre-period **RMSPE** is reported raw-daily (`norm_pre_rmspe`) and on the
  smoothed fit target (`norm_smooth_rmspe`, the headline). `poor_fit` is judged on the **smoothed**
  RMSPE (> 0.30). **Caveat kept deliberately:** on the *raw-daily* RMSPE the median is **0.332**
  and **177/300 events exceed 0.30** — SC cannot reproduce idiosyncratic daily noise, so SC fit
  must not be read as "good" on a raw-daily basis; the smoothed RMSPE is the credibility measure
  the flag uses.
- **Effect + CI.** `actual − synthetic` summed over promo and post windows; CI from the same
  moving-block residual bootstrap as ITS. **Caveat:** this bootstrap resamples the pre-period
  residual gaps only — it captures predictive noise but **not weight/model (donor-choice)
  uncertainty**, so the SC CIs are narrower than a fully honest interval. This is exactly why the
  placebo inference and credibility tiers below matter: they probe effect credibility in a way the
  bootstrap CI alone does not.
- **In-space placebo inference** (Abadie; `placebo_pvalue_promo/post`, `n_placebos`,
  `sc_low_credibility`). Each donor is re-fit as a *pseudo-treated* unit against the remaining
  donors (identical smoothing + level-fixed-effect procedure); its placebo effect is normalized by
  its own pre-period RMSPE (the standard ratio that corrects for fit quality), and the real effect's
  two-sided p-value is the rank of its RMSPE-normalized effect among all units
  (`(1 + #{placebo ratio ≥ real}) / (1 + n_placebos)`). An event whose promo-window placebo p-value
  exceeds `placebo_pvalue_threshold` (0.10) is flagged **`sc_low_credibility`** — its lift is not
  distinguishable from donor placebo noise. This is an **additive reported layer**; it does not
  change the point effects or the primary classification.

Results (`synth_results.parquet`): median smoothed normalized pre-RMSPE **0.112**; **39/300**
flagged poor-fit; median promotion-window effect +244 units; median promo-window **placebo
p-value 0.226**, and **195/300 events flagged `sc_low_credibility`** (p > 0.10). The metric is
well-behaved (Spearman correlation between a stronger RMSPE-normalized effect and a lower p-value
= −0.83; the 30 strongest lifts have median placebo p = 0.032). So the SC lifts are real on
average — and agree with ITS (§5) — but the *individual* promotion-window lift is often not
separable from placebo noise, which drives the credibility tiers in §7.

---

## 5. ITS vs synthetic control agreement (Key Question 4)

`effect_comparison.parquet` merges the two estimators. Restricting to the **203 events both
methods fit well** (`both_trustworthy`):

- **Promotion-window effect correlation 0.867.**
- Post-period effect correlation 0.444 (a smaller, noisier signal).
- Promotion-window **sign agreement 75.0%** across all 300 events; post-period 64.3%.
- Median promotion-window effect: ITS +228, SC +244 units.

Two independent counterfactuals agreeing strongly on the promotion-window lift is the confidence
signal for Key Question 4; divergence concentrates in the smaller post-period and in flagged
poor-fit events.

---

## 6. Cannibalization (`src/cannibalization.py`)

**Preferred estimate per event** (`chosen_method`): synthetic control where its pre-fit is good,
else ITS. On this run **SC 261 / ITS 39**. When **both** methods are poor-fit the event is still
computed but flagged **`low_confidence`** (32/300, including ITS-implausible events — such as
EVT00144 — when ITS is chosen).

With signed effects, a post-period dip is a negative `post_effect`:
- `cannibalization_units = max(0, −post_effect)` — the post dip in original units.
- `cannibalization_share = max(0, −post_effect) / promo_effect` when `promo_effect > 0`
  (0% = no pull-forward; ~100% = the entire gain was borrowed from the future). Left **un-capped**
  (a value above 1 means the post dip exceeded the lift).

Edge cases: `promo_effect ≤ 0` → share undefined (NaN) for 89/300 events (no positive lift to
borrow against); `post_effect > 0` → no cannibalization (units and share 0).

**Delisting/stockout caveat** (`post_near_zero`, threshold 20 units): a promotion whose total
post-window actual sales are at or below the threshold has a "dip" more likely from a **delisting
or stockout** than pull-forward; 11/300 events are flagged so their cannibalization is not
over-interpreted.

**Median cannibalization share (where defined) = 8.4%** — most promotions pull forward only a
small share of their lift; cannibalization alone is rarely the whole story. The discount cost does
most of the damage (§7).

---

## 7. Net impact and classification (`src/net_impact.py`)

**Incrementality asymmetry (the heart of Key Question 3).** The discount is paid on **every** unit
sold at the promo price, but margin is earned only on the **net incremental** units the promotion
created:

- `net_incremental_units = promo_effect + post_effect` (signed; equals
  `promo_lift − cannibalization_units` when the post window is a dip).
- `discount_cost = discount_depth × assumed_price × promo_units_sold`, where `promo_units_sold`
  is the **actual** units sold during the promo window — not the incremental ones.
- `incremental_margin = gross_margin × assumed_price × net_incremental_units`.
- `net_profit_impact = incremental_margin − discount_cost`.

Because the discount is charged on the full base but margin earned only on the incremental slice,
a promotion that lifts sales modestly can still lose money. `assumed_price = 1.0` → margin-units.

**Net-effect CI.** Propagated from the chosen method's promo + post percentile bounds by
**summing** them (`net_units_lo = promo_lo + post_lo`, `net_units_hi = promo_hi + post_hi`) — the
conservative, widest combination. The net-profit CI is the affine image (the discount cost is a
fixed offset).

**Classification.** `inconclusive` when the net-profit CI straddles zero; otherwise `net positive`
/ `net negative` by the sign of `net_profit_impact`. **`low_confidence` events are forced to
`inconclusive`** — a both-poor-fit estimate cannot support a confident call.

**Overall split (300 events; discount 0.20, margin 0.30, price 1.0):**

| Class | Count | % |
|-------|------:|---:|
| net positive | 8 | 2.7% |
| net negative | 210 | 70.0% |
| inconclusive | 82 | 27.3% |

Excluding the 32 low-confidence events: **8 / 210 / 50** (net negative 78.4%, net positive 3.0% of
the 268 confident events). **Median net profit −398.8 margin-units**; mean −764.9 (all) /
−505.5 (excluding low-confidence) — the all-events mean is pulled down by a few capped high-volume
low-confidence events (now inconclusive), so the headline rests on the **median and the confident
subset**, not the mean.

**Credibility tiers (reported, primary labels unchanged).** Because 195/300 individual SC lifts are
not separable from placebo noise (§4), the confident labels are stratified by whether the *chosen*
estimate is itself credible — a chosen SC with `sc_low_credibility == False`, or a chosen ITS with
`poor_fit == False`:

- **Net negative (210): 68 high-confidence / 142 lower-confidence.**
- **Net positive (8): 8 high-confidence / 0 lower-confidence.**

Two robustness reads bracket the net-negative count: the **conservative lower bound** (downgrade
every low-credibility SC lift to inconclusive) gives **8 / 68 / 224**; and a **placebo-calibrated**
check finds **140 of the 142** lower-confidence net-negatives stay net-negative under
90th-percentile placebo noise (109 even under the maximum placebo). So the net-negative direction is
strong but credibility-graded, not a flat "70% lose money."

**Key Question 3:** **211** promotions are naive wins (positive promotion-window lift), of which
**151 flip to net negative** after cannibalization + discount — the central finding.

**Break-even (dependence on the modeled economics).** Per event the break-even discount is
`gross_margin × net_incremental_units / promo_units_sold` (the discount at which net profit is
zero, margin held fixed) and the break-even margin is `discount_depth × promo_units_sold /
net_incremental_units`. On this run **127/300 never break even** (net incremental units ≤ 0); among
positive-lift events the **median break-even discount is 7.5%** and the **median break-even margin
is ~80%** — i.e. the typical promotion is underwater at the assumed 20%/30% because the discount is
spread over units that would have sold anyway. **Caveat:** the summary counts "would flip net
positive at a 10% discount (43 events) / a 40% margin (18 events)" are **point-estimate flips that
hold the measured lift fixed**; a shallower discount would in reality likely produce a *smaller*
lift, so these are optimistic upper bounds, not forecasts.

**Sensitivity grid** (median net profit / % net negative per cell):

| discount \ margin | 0.20 | 0.30 | 0.40 |
|---|---|---|---|
| **0.10** | −185.7 / 59.7% | −157.1 / 44.7% | −132.6 / 34.0% |
| **0.20** | −431.8 / 81.7% | **−398.8 / 70.0%** | −371.5 / 59.7% |
| **0.30** | −674.3 / 86.7% | −647.7 / 81.7% | −620.7 / 75.0% |

The direction is **robust across the whole grid** (net-negative-dominant in every cell, ~34%–87%).
Even the most generous assumptions (10% discount, 40% margin) leave 34.0% net negative vs 12.3%
net positive. The verdict is driven far more by **discount depth** than by margin.

---

## 8. Early-warning rule and backtest (`src/early_warning.py`)

**Signal** (first `early_days` = 3 promotion days only; no day-4-onward data): the chosen method's
daily lift `actual − cf` over the first 3 days, averaged and **normalized by the pre-period mean
daily sales**, so high- and low-volume item-stores are comparable. A low signal means the
promotion is barely lifting relative to its own baseline.

**Shared-counterfactual note.** The early signal is built from the first 3 days of the *same* chosen
counterfactual used to compute the full-window label, so signal and label share those 3 days. This
is a built-in correlation, not look-ahead: the signal still uses **no** data from day 4 onward, and
the backtest predicts the full-window outcome from the early slice.

**Label:** `1 = net negative`, `0 = not`. **Inconclusive promotions are grouped with label 0**
(the rule's job is to catch *confidently* net-negative promotions); 82 inconclusive events are
grouped this way.

**Backtest = stratified 5-fold cross-validation** (`k_folds = 5`, stratified on the net-negative
label, seeded; folds disjoint and covering every scored event, enforced and unit-tested). For each
fold the threshold is tuned on the other folds and evaluated on the held-out fold, so every one of
the 300 events gets exactly one **out-of-fold** prediction. The CV is run under **both** tuning
criteria and both are reported; the deployable metric is chosen by whichever gives a real pooled
precision lift over the base rate at usable recall (`tune_metric: "auto"`).

| Metric | deployable threshold | precision (mean ± std) | recall (mean ± std) | F1 (mean ± std) | pooled OOF precision / recall |
|---|--:|--:|--:|--:|--:|
| **Youden's J** (deployed) | 0.3664 | 0.781 ± 0.043 | 0.867 ± 0.032 | 0.820 ± 0.025 | **0.778 / 0.867** |
| F1 (alternative) | 0.8432 | 0.732 ± 0.025 | 0.986 ± 0.012 | 0.840 ± 0.020 | 0.731 / 0.986 |

Pooled out-of-fold confusion (Youden, n = 300): TP 182, FP 52, FN 28, TN 38. Base rate 0.70.

**Honest base-rate comparison.** The sample is imbalanced (70% net negative), so the trivial "flag
everything" rule already scores precision 0.70 at recall 1.0. **F1 tuning is near-trivial** — it
drives the threshold to flag almost everything (precision 0.731 ≈ base rate). **Youden's J is the
deployable choice:** pooled precision **0.778 vs the 0.70 base rate is a real but modest lift
(+0.078)** at recall 0.867. Two honesty caveats remain: the metric was selected on the CV (not in a
nested loop), a mild optimism; and the sample is small, so the ±std figures are wide.

**Failure cases (pooled out-of-fold):**
- *False negatives* (missed losers) carry a strong early signal yet end net negative (e.g.
  EVT00055 signal 0.63; EVT00042 signal 1.15) — they lift strongly in the first 3 days but lose on
  later cannibalization + discount, which the early rule cannot see by construction.
- *False positives* are largely events the full analysis classifies **inconclusive** with low early
  signals — borderline, not clear mistakes.

---

## 9. Dashboard (`src/build_dashboard.py`)

A single **self-contained static page** (`docs_site/index.html`) with all per-event results
precomputed and embedded as inline JSON — no backend, no network calls at view time, works on
GitHub Pages and from `file://`. Plotly is vendored locally (`docs_site/assets/plotly.min.js`,
pinned v2.27.0, referenced by relative path). Two views: an **Overview** (classification split, the
credibility tiers + conservative bound, break-even summary, Key-Question-3 headline, sensitivity
grid, the 5-fold cross-validated early-warning panel with the base-rate caveat, and labeled economic
assumptions) and an **Explore** view (searchable selector over all 300 promotions; per-promotion
chart with actual + ITS + SC counterfactuals and the promotion window shaded, plus the net-impact
components, classification badge, chosen method, fit quality, SC placebo p-values and credibility,
break-even discount/margin, and any caution flags). First-draw chart resize is handled on tab
reveal so the chart tracks the container at any width.

---

## 10. Caveats (carried throughout)

- Economics are **modeled assumptions**, not data; the sensitivity grid and break-even analysis are
  the honest read on how far the conclusions depend on them. "Most promotions lose money" is
  conditional on the 20% discount / 30% margin assumptions.
- All effect CIs are **percentile** bootstrap intervals; the net CI is the conservative sum of the
  chosen method's promo + post bounds. The SC bootstrap omits weight/model uncertainty, so the
  placebo inference + credibility tiers (not the CI alone) carry the per-event credibility story.
- Most *individual* SC lifts are not separable from donor placebo noise (195/300 `sc_low_credibility`),
  which is why the net-negative verdict is credibility-tiered (68 high / 142 lower of 210) rather
  than taken at face value; 140/142 survive 90th-percentile placebo-scale optimistic noise (109/142
under the maximum placebo).
- SC fit is weak on **raw-daily** RMSPE (median 0.332); the credibility flag uses the smoothed
  RMSPE, and SC must not be presented as a good raw-daily match.
- The ITS plausibility cap bounds exponential-extrapolation blow-ups (15 events); its 6× mean
  threshold was tuned post-hoc on this run and should be re-checked on new data. Only EVT00144 was a
  newly-caught good-fit event (now low-confidence → inconclusive); the rest were already poor-fit.
- The break-even "would flip positive" counts hold the measured lift fixed at a different discount,
  which is optimistic (a shallower discount would likely shrink the lift).
- `post_near_zero` events may be delistings/stockouts rather than cannibalization.
- The early-warning backtest is a **small, imbalanced** sample (n = 300, 70% net negative); 5-fold
  CV reduces but does not eliminate the fragility, the metric is chosen on the CV (not nested), and
  the signal shares the first 3 days of the label's counterfactual. Metrics are reported against the
  base rate and not overclaimed.
