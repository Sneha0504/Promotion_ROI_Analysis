# Findings memo — Promotion ROI Analysis

Internal memo for the project owner. Every number is from the committed pipeline outputs in
`data/processed/` (`make all`, run 2026-11-04). Not linked from the public README.

## The question

For a sample of real retail promotions, did they actually make money — after accounting for the
sales that would have happened anyway, for demand pulled forward from the future
(cannibalization), and for the cost of the discount itself? A promotion that lifts sales during
its window can still be a net loss once those three things are netted out.

## What was done

- Selected **300** clean promotion events (34 stores, 81 items, 7 families, 2014–2017) from the
  ~125M-row Corporación Favorita dataset, by an explicit, documented funnel — not cherry-picked.
- Estimated each promotion's causal effect with **two independent counterfactuals**: an interrupted
  time series (ITS) regression and a synthetic control (SC) built from comparable non-promoted
  item-stores. Reported effects over the promotion window and the following 28 days, each with a
  percentile-bootstrap confidence interval.
- Combined lift, cannibalization, discount cost and margin into a **net profit impact** per
  promotion and classified each as net positive, net negative or inconclusive.
- Built a simple **early-warning rule** (first 3 days only) and backtested it with 5-fold
  stratified cross-validation.
- Published a precomputed, self-contained interactive dashboard.

## Headline findings

- **The two methods agree on the lift** (correlation **0.867** on the 203 promotions both fit well),
  the main confidence signal — but individually, **195/300 synthetic-control lifts are not separable
  from donor placebo noise**, so the verdict is credibility-tiered, not taken at face value.
- **Under the assumed economics, most promotions lose money — credibility-tiered.** 300 total,
  primary split **8 net positive (2.7%) / 210 net negative (70.0%) / 82 inconclusive (27.3%)**. Of
  the 210 net-negative, **68 are high-confidence / 142 lower-confidence**; **140 of the 142 survive
  90th-percentile placebo-scale optimistic noise** (109 under the maximum placebo), and the
  conservative lower bound (all low-credibility SC →
  inconclusive) is **8 / 68 / 224**. **Median net profit −399 margin-units** (mean −765; the median
  leads). The **8 net-positive are all high-confidence**.
- **"Lose money" is conditional on the assumptions.** By break-even, the median positive-lift
  promotion would need a discount **below ~7.5%** or a margin **near ~80%** to break even at the
  assumed 20%/30%; 127 never break even at any discount.
- **"Wins" that lose:** **211** promotions look like wins on a naive before/after view; **151 of
  them (72%) are net negative** after cannibalization + discount. This is the core business point.
- **Direction robust to assumptions:** across the discount × margin sensitivity grid the
  net-negative share is ~34%–87% and dominates every cell; the verdict is driven mainly by discount
  depth.
- **Cannibalization is modest** (median share 8.4% of lift, where defined) — the discount cost on
  would-have-sold-anyway units does most of the damage, not pulled-forward demand.
- **Early warning:** under 5-fold cross-validation (pooled out-of-fold, n = 300) the deployable rule
  (Youden's J, threshold 0.3664) gives **precision 0.778 / recall 0.867 vs a 0.70 base rate** — a
  real but modest lift. (Tuning on F1 is near-trivial — it flags almost everything.) Useful as a
  triage signal, not a precise predictor.

## Key judgment calls

- **Economics are modeled assumptions, not data.** Favorita has no price or margin, so price was
  normalized to 1.0 (results in margin-units), discount depth set to 20% and gross margin to 30%.
  The sensitivity grid exists precisely to show the net-negative conclusion survives a range of
  these values.
- **Discount cost is charged on all promo units, margin earned only on incremental units** (the
  incrementality asymmetry). This is what makes a modest-lift promotion a net loss, and it is the
  main driver of the headline.
- **Sample selection** enforced clean windows (no overlapping promotions), ≥ 90 days of history and
  store-open days — favoring credible estimates over sample size.
- **Synthetic control variant:** weights are fit on a 7-day-smoothed pre-period with a pre-period
  level fixed effect, because high-volume targets sit outside the convex hull of their smaller
  same-family donors. This is a documented deviation from textbook SC (available via a config flag).
- **Method preference:** SC is used where its pre-fit is good (261 events), ITS otherwise (39);
  events where both fit poorly (32) are forced to inconclusive so a weak estimate never gets a
  confident label.
- **Credibility is reported, not buried:** in-space placebo inference flags the SC lifts that are
  not separable from donor noise, and the net-negative verdict is stratified into high- (68) vs
  lower-confidence (142) tiers rather than presented as a flat count.

## Limitations

- **No price/margin in the data** — profit figures are assumption-driven. The break-even analysis
  quantifies the dependence: the median positive-lift promotion needs a discount below ~7.5% or a
  margin near ~80% to break even. (The "would flip positive at a shallower discount" counts hold the
  lift fixed, which is optimistic — a shallower discount would likely shrink the lift.)
- **Most individual SC lifts are not separable from placebo noise** (195/300); the headline rests on
  the credibility tiers and the placebo-robustness check (140/142 survive 90th-percentile
  placebo-scale noise; 109/142 under the maximum placebo), not on the raw 210 count. The SC bootstrap CIs also omit weight/model uncertainty, so they are
  narrower than a fully honest interval — another reason the placebo layer matters.
- **SC fit is weak on raw-daily sales** (median raw-daily RMSPE 0.332); the credibility flag uses
  the smoothed fit, and SC should not be read as a tight day-by-day match.
- **A few ITS counterfactuals extrapolate explosively** (exponential back-transform of a log-linear
  trend); 15 are capped and flagged by the two-criterion guard (max- and mean-based). Only EVT00144
  was a newly-caught good-fit event (now low-confidence → inconclusive); the mean multiple (6×) was
  tuned post-hoc on this run and should be re-checked on new data.
- **Some large post-period "dips" may be delistings or stockouts**, not cannibalization; these are
  flagged (`post_near_zero`) and should not be over-interpreted.
- **The early-warning backtest is small and imbalanced** (n = 300, 70% net negative); 5-fold CV
  reduces but does not eliminate the fragility, the tuning metric was chosen on the CV (not nested),
  and the early signal shares the first 3 days of the label's counterfactual.
