# PLAN.md — Promotion_ROI_Analysis

Internal working document. Not linked from the public README. This is the design I
will build against; it is the approval gate before any analysis code is written.

---

## 1. Goal and key questions

**Goal.** Measure whether real retail promotions actually *made money*, not just
whether sales rose during the promotion window. For each analyzed promotion, isolate
its true causal effect from the trend, seasonality, and noise that would have
happened anyway (using interrupted time series **and** synthetic control — never a
naive before/after), quantify cannibalization (pulled-forward demand that depresses
post-promotion sales), and compute a net profit impact after the discount cost.
Ship the results as a precomputed interactive dashboard on GitHub Pages.

**Key questions** (restated):
1. Each promotion's TRUE sales effect, isolated from trend/seasonality/noise.
2. How much of that effect is cannibalization (a post-promotion dip).
3. After cannibalization + discount cost, which promotions were net positive, net
   negative, or inconclusive — including ones that look like wins naively but lose.
4. Do ITS and synthetic control agree? Where they diverge, what does that say about
   confidence?
5. Can a simple rule, checked a few days in, flag a likely net-negative promotion
   early? How well does it backtest?

---

## 2. The data and why it fits

**Corporación Favorita Grocery Sales Forecasting** (Kaggle competition). Already
downloaded locally and extracted to `data/raw/` (git-ignored). Confirmed contents:

| File | Rows (excl. header) | Key columns |
|------|--------------------:|-------------|
| `train.csv` | ~125M (extracting) | `id, date, store_nbr, item_nbr, unit_sales, onpromotion` |
| `items.csv` | 4,100 | `item_nbr, family, class, perishable` |
| `stores.csv` | 54 | `store_nbr, city, state, type, cluster` |
| `oil.csv` | 1,218 | `date, dcoilwtico` (daily WTI oil price) |
| `holidays_events.csv` | 350 | `date, type, locale, locale_name, description, transferred` |
| `transactions.csv` | 83,488 | `date, store_nbr, transactions` (per-store daily txn count) |

**Why it fits.** It is real daily unit-level demand with an explicit per-item-per-day
`onpromotion` flag across many stores and product families — exactly the structure
needed to (a) define real promotion events, (b) build long pre-periods for
counterfactual modeling, (c) assemble donor pools of comparable non-promoted
item-stores for synthetic control, and (d) observe the post-promotion window for
cannibalization. Holidays and oil give external regressors/closure cross-checks.

**Important data limitation (drives a methodology decision).** Favorita contains
**no price and no margin** data anywhere. So "discount cost" and "net profit" cannot
be read off the data — they must be modeled with clearly labeled, configurable
assumptions (discount depth, gross margin). See §6. This is called out honestly in
the dashboard, README, and `docs/memo.md`.

**Scale handling.** `train.csv` is ~4.6 GB / ~125M rows. It is **never** loaded
whole. `src/load.py` reads it in chunks (pandas `chunksize`) with a dtype schema and
filters to only the (store_nbr, item_nbr) pairs and date ranges needed for the
selected promotion events + their donor pools, writing a compact Parquet panel. The
exact filter is documented in `docs/data_sources.md`.

---

## 3. Unit of analysis and sample selection

- **Promotion event** = a maximal run of consecutive days with `onpromotion = True`
  for one `(item_nbr, store_nbr)`, tolerating gaps of ≤1 day (so a flickering flag
  is treated as one event; flickering is logged as a data-quality rule).
- **Windows** (config-driven): `pre_days = 90`, `post_days = 28`, promotion window =
  the event's own length.
- **Selection criteria** (config-driven, finalized after profiling in Step 4):
  - ≥ `pre_days` of pre-period history for the item-store, with sufficient non-zero
    sales (avoid near-all-zero series that can't be modeled).
  - Promotion length ≥ `min_promo_days = 3` and ≤ a cap (exclude permanent/always-on
    flags).
  - Clean windows: the same item-store is **not** on another promotion inside its
    pre/post windows (so post-period isn't contaminated by the next promo).
  - A usable donor pool exists (see §5). *(Deferred to `build_panel` in the next unit;
    the Unit-1 `load.py` does not yet enforce this, so the selected count may drop when
    donor pools are built.)*
  - Store open across the windows (closure days handled per §4).
- **Target sample size:** on the order of 100–300 analyzable promotions. The exact
  count is reported after profiling; selection is by explicit rule, not cherry-picked.

---

## 4. Data quality (→ `docs/data_quality_report.md`)

Each check becomes a numbered rule with severity + handling. At minimum:
1. Missing/negative `unit_sales` (Favorita uses negatives for returns) — handling:
   clip or treat per rule; documented.
2. Missing `onpromotion` flags (the flag is null in the early part of the series) —
   restrict the analysis window to the period where the flag is reliably populated.
3. Items/stores with too-short history — excluded by the pre-period rule (§3).
4. Zero-sales days that are **store closures** vs genuine zero demand — cross-check
   against `holidays_events.csv` and `transactions.csv` (a store with 0 transactions
   that day is closed); closure days are masked, not treated as zero demand.
   *(Unit-1 `load.py` uses only the transactions-span check for store-open; the
   holiday-closure cross-check and interior-closure masking are deferred to
   `build_panel` in the next unit.)*
5. Oil price coverage gaps — forward-fill documented; oil used only as an
   optional regressor.
6. Flickering on/off promotion flags within one event — merged via the ≤1-day-gap
   rule; count reported.

Output also records the exact sample-selection criteria used and the resulting count
of analyzable promotions.

---

## 5. Analysis panel (`src/build_panel.py`)

Per selected promotion event, build a tidy daily panel over
`[pre-period, promotion window, post-period]` containing the promoted item-store's
`unit_sales`, plus the same daily series for each **donor** item-store.

**Donor-pool definition ("similar", documented exactly):**
- Same product **family** as the promoted item (primary similarity axis), optionally
  same `class`.
- Drawn from the **same store** first (different items), then expanded to stores of
  the same `cluster`/`type` if more donors are needed.
- Donor must **not** be on promotion at any point during the target's combined
  pre+promo+post window (clean controls).
- Donor must have complete, non-degenerate sales over the window.
- Cap the pool size (e.g. top N by pre-period correlation with the target) for a
  well-posed synthetic-control fit.

---

## 6. Methods

### 6a. Interrupted time series (`src/interrupted_ts.py`)
- Fit on the **pre-period only**: `unit_sales` (likely `log1p`-transformed) ~
  linear time trend + day-of-week dummies + holiday indicator (+ optional oil
  regressor). Start with OLS/segmented regression; SARIMA as a documented fallback
  if residual autocorrelation is severe.
- Project the fitted model forward through the promotion and post-period as the
  counterfactual ("what would have happened without the promotion").
- Effect = actual − projected, summed over the promotion window and (separately) the
  post-period.
- Report a **confidence interval** (prediction interval / residual bootstrap), not
  just a point estimate.
- **Diagnostics:** pre-period fit R²/RMSE and residual checks (Durbin–Watson /
  Ljung–Box). Promotions whose pre-period fit is too poor are **flagged/excluded**.

### 6b. Synthetic control (`src/synthetic_control.py`)
- Build a weighted combination of donors matching the target's **pre-period** daily
  pattern, weights **non-negative and summing to 1** (standard SC), solved by
  constrained optimization (SciPy) / non-negative least squares.
- Report **pre-period fit quality explicitly** (pre-period RMSPE) — this is the
  credibility check for the whole method; poor pre-fit is flagged.
- Effect = actual − synthetic through the promotion and post-period.
- **Compare to ITS** for the same promotions: report where they agree and where they
  diverge, and treat divergence as a confidence signal (Q4).

### 6c. Cannibalization (`src/cannibalization.py`)
- Preferred effect estimate = **synthetic control where its pre-fit is good**, else
  fall back to **ITS** (noted per promotion).
- Cannibalization = the post-period cumulative dip expressed as a **share of the
  promotion-window cumulative lift** (0% = no pull-forward; ~100% = the gain was
  entirely borrowed from the future).

### 6d. Net impact (`src/net_impact.py`)
- Net incremental units = promotion-window lift − post-period cannibalization.
- **Discount cost & margin are modeled assumptions** (no price in data). In
  `config.yaml`: `assumed_price` (normalized to 1.0 by default so results read as
  margin-units), `discount_depth` (e.g. 0.20), `gross_margin` (e.g. 0.30).
  - `discount_cost = discount_depth * assumed_price * promo_units_sold`
  - `incremental_margin = gross_margin * assumed_price * net_incremental_units`
  - `net_profit_impact = incremental_margin − discount_cost`
- Each assumption is **clearly labeled** on the dashboard/README; a short
  sensitivity note shows how the net sign moves with discount depth / margin.
- **Classification** per promotion: **net positive**, **net negative**, or
  **inconclusive** (effect CI straddles zero / wide). Report the overall split.

### 6e. Early-warning rule + backtest (`src/early_warning.py`)
- Using **only the first `early_days` (e.g. 3) days** of the promotion, compute an
  early-lift signal relative to the counterfactual / the item's historical promo
  response.
- Simple, explainable **threshold rule** flags promotions likely to end net negative.
- **Backtest:** tune the threshold on a train split, evaluate on a **disjoint
  held-out** set of promotions; report **precision and recall honestly**, including
  failure cases. Train/test disjointness is enforced and unit-tested (Step 10).

---

## 7. Dashboard (`src/build_dashboard.py` → `docs_site/index.html`)
- Single **self-contained static HTML** (Plotly), **all results precomputed and
  embedded as JSON** — no server, no API at view time (GitHub Pages serves static
  files only). Must open correctly as a local `file://`.
- Searchable **dropdown** to pick any analyzed promotion (by item + store).
- Per-promotion chart: **actual** sales, **ITS counterfactual**, **synthetic-control
  counterfactual** on one legended chart, with the **promotion window shaded**.
- Net impact number + components (lift, cannibalization, discount cost) + the
  classification.
- **Summary view:** overall classification split + the early-warning rule's backtest
  precision/recall.
- Embedded data is the summarized per-promotion results sample (not raw
  transactions), kept small for fast load.
- Published via **GitHub Pages** (main branch, `/docs_site`); final live URL recorded.

---

## 8. Tests (`tests/`, pytest) — see Step 10
ITS counterfactual matches a hand-calculable value on a synthetic linear-trend series
with a known injected effect; SC weights are non-negative and sum to 1; the pre-fit
diagnostic flags a deliberately poor SC match; net impact is internally consistent
(lift − cannibalization − discount = reported net) on a constructed example; the
early-warning backtest keeps train/tune and held-out sets disjoint.

## 9. Code standards — see Step 11
Python 3.11+ (using 3.12.6), pinned `requirements.txt`, venv, type hints, docstrings,
`logging` (not print), config in `config/config.yaml`, no hard-coded paths/creds,
`ruff` lint. Makefile targets: `setup, download, panel, its, synth, impact, warning,
test, dashboard, all`.

---

## 10. Risks and fallbacks
| Risk | Fallback |
|------|----------|
| No price/margin in the dataset | Model discount & margin as labeled, configurable assumptions; report in margin-units + sensitivity. |
| `onpromotion` null in early series | Restrict analysis to the window where the flag is reliably populated. |
| Sparse/zero-heavy daily item-store series | Favor higher-volume families; enforce non-zero-sales selection rule; relax/aggregate if sample too small. |
| Synthetic control pre-fit poor | Fall back to ITS for that promotion, flagged on the dashboard. |
| 125M-row train.csv won't fit in memory | Chunked, filtered read to a compact Parquet panel; documented. |
| Too few clean promotions after filtering | Loosen non-critical criteria (donor pool size, post window) within documented bounds. |
| Tight disk (~11 GB free) | Keep only the filtered panel long-term; raw train.csv git-ignored and can be re-extracted. |

## 11. Out of scope
Forecasting future sales (the original Kaggle task); true margin/price reconstruction;
cross-item substitution beyond same-item cannibalization; store-level assortment
effects; real-time/productionized serving; causal methods beyond ITS + synthetic
control.

---

## Build order (after approval)
1. Config + `load.py` (chunked filtered extraction) + `data_sources.md`.
2. Profiling + `data_quality_report.md` + finalize selection criteria.
3. `build_panel.py` (panels + donor pools).
4. `interrupted_ts.py`; 5. `synthetic_control.py`; 6. `cannibalization.py` +
   `net_impact.py`; 7. `early_warning.py`.
8. Tests (pytest) + ruff clean.
9. `build_dashboard.py` → `docs_site/index.html`; publish GitHub Pages.
10. `README.md` (strict public rules), `memo.md`, `methodology.md`, `AI_USE.md`.
11. `make all` from clean; verify every README/memo number against script output;
    scan repo + commits for any attribution markers; final summary.

Each coding unit is implemented, then independently reviewed for correctness, repeating until it passes.
