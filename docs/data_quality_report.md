# Data quality report

All numbers below were produced by scanning the raw data with DuckDB + pandas
(`src/load.py` / `sql/profile_train.sql`); the headline counts are also persisted to
`data/processed/selection_stats.json`. Profiling run date: 2026-11-04.

## Headline profile of `train.csv`
Single streaming scan of all **125,497,040** data rows (dates 2013-01-01 … 2017-08-15):

| Metric | Value |
|--------|------:|
| Rows | 125,497,040 |
| Distinct stores | 54 |
| Distinct items (present in train) | 4,036 |
| `unit_sales` NULL | 0 |
| `unit_sales` < 0 (returns) | 7,795 |
| `unit_sales` = 0 (explicit zero rows) | 0 |
| `onpromotion` NULL | 21,657,651 |
| `onpromotion` = True | 7,810,622 |
| `onpromotion` = False | 96,028,767 |
| First non-null `onpromotion` date | 2014-04-01 |

Supporting files: `items.csv` lists 4,100 items (64 never appear in train); `oil.csv`
1,218 rows (2013-01-01 … 2017-08-31) with 43 null prices; `holidays_events.csv` 350 rows
(types: Holiday 221, Event 56, Additional 51, Transfer 12, Bridge 5, Work Day 5; 12
flagged `transferred`); `transactions.csv` 83,488 rows with **0** rows where
`transactions = 0`.

## Numbered data-quality rules

1. **Negative `unit_sales` (returns).** Severity: medium. 7,795 rows have negative
   `unit_sales` (Favorita encodes returns as negatives); there are **0** null sales.
   Handling: kept in extraction; at modeling time negative daily values are clipped to
   0 (or handled per the model's documented rule) so returns do not create artificial
   demand. Counted, not silently dropped.

2. **`onpromotion` NULL early in the series.** Severity: high. 21,657,651 rows have a
   null promo flag, and the first non-null flag is exactly **2014-04-01**. Handling: the
   analysis is restricted to `date >= reliable_flag_start` (`2014-04-01`, in
   `config/config.yaml`); earlier rows are never used to detect or window promotions.

3. **Sparse storage — zero-demand days are absent, not zero.** Severity: high
   (methodological). `train.csv` contains **0** explicit zero-sales rows: a
   (store, item) with no sale on a day simply has no row that day. Consequence: a
   series-level non-zero fraction computed over *stored* rows is ~1.0 for essentially
   every item-store (observed min across selected events: 0.99). The
   `min_pre_nonzero_frac` selection rule therefore acts only as a guard against
   degenerate all-negative/return series and barely binds here; **true calendar-day
   density** (treating absent days as zero demand) is assessed when the dense daily
   panel is built in the next unit, where near-all-zero pre-periods are dropped.

4. **Store closures vs genuine zero demand.** Severity: medium. `transactions.csv` has
   **0** rows with `transactions = 0`; a closed store-day is represented by an *absent*
   row, and 53 of 54 stores have their first transaction after 2013-01-01 (staggered
   store openings). Handling: the store-open selection rule requires each store's
   transaction span (first…last transaction date) to cover the event's full
   `[pre_start, post_end]` window, which also enforces a complete post-window within the
   data horizon (last transaction date 2017-08-15). Interior closure days are masked
   (not treated as zero demand) when the daily panel is built.

5. **Oil price coverage gaps.** Severity: low. `oil.csv` has 43 null `dcoilwtico`
   values and is quoted on business days only (no weekend rows), though no business day
   is missing within its span. Handling: forward-fill when used; oil is only an optional
   regressor, never a filter.

6. **Flickering on/off promotion flags.** Severity: low. Of 3,455,720 raw promoted-day
   runs, **286,616 (~8.3%)** contain at least one merged intra-run gap (the flag
   flickered off for ≤ `promo_flag_gap_tolerance` = 1 day). Handling: such gaps are
   merged so one flickering promotion is treated as a single event
   (`promo_len` > `n_promo_days` marks these).

7. **Items/stores with too-short history.** Severity: medium. 9,958 of 139,364 promoted
   item-store series have fewer than `pre_days` = 90 observed days. Handling: excluded by
   the pre-period history rule (event dropped unless the series is observed from on/before
   `pre_start`, within the reliable-flag window).

8. **Items in metadata but not in sales.** Severity: low. `items.csv` lists 4,100 items
   but only 4,036 appear in `train.csv` (64 never sold in the period). Handling: metadata
   is left-joined; events are only ever formed from items present in the sales data.

## Exact selection criteria used
From `config/config.yaml` (`selection` + `windows`), applied in `src/load.py`:

- Promotion event = maximal run of consecutive `onpromotion = True` days per
  (store_nbr, item_nbr), merging gaps of ≤ `promo_flag_gap_tolerance` = 1 day.
- `min_promo_days` = 3 ≤ promo length ≤ `max_promo_days` = 60 (calendar span).
- ≥ `pre_days` = 90 days of pre-period history within the reliable-flag window
  (series observed from on/before `pre_start`, and `pre_start` ≥ 2014-04-01).
- Series non-zero fraction ≥ `min_pre_nonzero_frac` = 0.5 (series-level proxy; see
  rule 3).
- Clean window: no *other* promotion of the same item-store overlaps
  `[pre_start, post_end]` (`post_days` = 28).
- Store open across the window (rule 4).
- Deterministic cap to `max_events` = 300 by **descending pre-period-only sales volume**
  — `SUM(unit_sales)` over each event's `[pre_start, promo_start)` window (computed in
  DuckDB, `sql/pre_period_volume.sql`). This key is strictly pre-intervention: it ends
  the day before the promotion starts and contains no promotion-period or post-period
  days, so it is genuinely outcome-independent. Tie-broken by store/item/date — the
  sample is **not** cherry-picked by promotion outcome.

## Selection funnel (real counts)
From the production run (`data/processed/selection_stats.json`):

| Stage | Events remaining |
|-------|-----------------:|
| Raw detected events | 3,455,720 |
| After promo-length bounds (3–60d) | 427,674 |
| After pre-period history | 405,849 |
| After non-zero density proxy | 405,849 |
| After clean (non-overlapping) window | 46,728 |
| After store-open check | 42,807 |
| **Final (capped at `max_events` = 300)** | **300** |

**Resulting analyzable sample: 300 promotions** (the cap; 42,807 qualified before the
cap). Promotions in the sample start between 2014-08-07 and 2017-06-21; the full
pre/post analysis windows span 2014-05-09 (earliest `pre_start`) to 2017-08-09 (latest
`post_end`, within the 2017-08-15 data horizon). The sample covers 34 distinct stores and
81 distinct items. Because the cap ranks by pre-period sales volume, the sample skews to
high-volume families (BEVERAGES 198, GROCERY I 83, BREAD/BAKERY 14, CLEANING 2,
LIQUOR/WINE/BEER 1, PERSONAL CARE 1, FROZEN FOODS 1); this is a deterministic, documented
consequence of volume-ranked capping, not outcome selection. Promotion length ranges
3–56 days (median 21). The ranking key `pre_period_unit_sales` ranges from 5,954 to
55,971 units (median 7,894) across the selected events.
