# Data sources

## Dataset
**Corporación Favorita Grocery Sales Forecasting** (Kaggle competition).

- URL: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data
- Date accessed: 2026-11-04
- License / terms: Kaggle competition data, governed by the competition rules. It is
  used here for non-commercial analysis purposes. The raw CSVs are **not**
  redistributed in this repository — they are git-ignored (`data/raw/`) and must be
  downloaded from Kaggle under the competition's terms (accepting the rules is required
  before download).

## Files and verified row counts
Row counts are data rows (excluding the header) and were produced by scanning the files
with DuckDB; where a count differs from a commonly quoted figure the difference is
explained.

| File | Rows (data) | Key columns |
|------|------------:|-------------|
| `train.csv` | 125,497,040 | `id, date, store_nbr, item_nbr, unit_sales, onpromotion` |
| `items.csv` | 4,100 | `item_nbr, family, class, perishable` |
| `stores.csv` | 54 | `store_nbr, city, state, type, cluster` |
| `oil.csv` | 1,218 | `date, dcoilwtico` |
| `holidays_events.csv` | 350 | `date, type, locale, locale_name, description, transferred` |
| `transactions.csv` | 83,488 | `date, store_nbr, transactions` |

Note on `train.csv`: the `id` column runs 0 … 125,497,039, so there are 125,497,040
data rows (the figure 125,497,039 sometimes quoted is the maximum `id`, not the row
count). Confirmed by `COUNT(*)` in DuckDB.

## Date range
- `train.csv`: **2013-01-01 through 2017-08-15** (start confirmed via `MIN(date)` in SQL).
- `transactions.csv`: 2013-01-01 through 2017-08-15 (matches train).
- `oil.csv`: 2013-01-01 through 2017-08-31 (extends slightly past the sales horizon).

Favorita has **no price and no margin columns** in any file; discount cost and net
profit are therefore modeled with labeled assumptions (see `config/config.yaml`
`economics` and the methodology notes). The `onpromotion` flag is stored as text
(`True`/`False`) and is **NULL in the early part of the series**; the analysis is
restricted to the window where it is reliably populated (see
`docs/data_quality_report.md`).

## Extraction / chunking approach (how scale is handled)
`train.csv` is ~4.7 GB / ~125M rows and is **never loaded whole into memory**. All
extraction and aggregation is done with **DuckDB SQL** run directly against the CSVs via
`read_csv_auto`, which scans the file as a stream (the DuckDB connection is bounded with
`PRAGMA threads` and `PRAGMA memory_limit`). The `.sql` query files live in `sql/` and
are executed from `src/load.py`:

- `sql/profile_train.sql` — one-pass data-quality profile of `train.csv`.
- `sql/candidate_series.sql` — per `(store_nbr, item_nbr)` promoted-day / non-zero-day
  counts, date span and total sales (within the reliable-flag window), used to shortlist
  series.
- `sql/promoted_rows.sql` — all promoted `(store_nbr, item_nbr, date)` rows (the
  promoted subset is small enough to detect consecutive-day runs in pandas).
- `sql/pre_period_volume.sql` — per candidate event, `SUM(unit_sales)` over its
  `[pre_start, promo_start)` pre-period window; the outcome-independent ranking key used
  to cap the sample to `max_events`.
- `sql/daily_panel.sql` — the daily `unit_sales` panel for a given set of item-stores
  and date range (consumed by the later panel builder).

The compact derived outputs are written to `data/processed/` and **are committed to the
repository** (they are small — a few tens of KB): `selected_events.parquet` (+ a `.csv`
copy) and `selection_stats.json`. The raw `train.csv` is not committed and can be
re-downloaded and re-extracted from Kaggle at any time; re-running `make select`
regenerates the processed outputs. DuckDB spill, if any, goes to the git-ignored
`data/interim/` scratch directory and is cleaned up after each run.
