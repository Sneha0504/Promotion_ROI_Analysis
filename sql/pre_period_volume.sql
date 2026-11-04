-- pre_period_volume.sql
-- Pre-period-only sales volume per candidate event: SUM(unit_sales) over the event's
-- [pre_start, promo_start) window. This is the OUTCOME-INDEPENDENT ranking key for the
-- max_events cap: the window ends strictly before the promotion begins, so it contains
-- no promotion-period or post-period days and cannot leak the promotion's response.
-- Runs as a streaming scan of train.csv joined to the (small) candidate-event relation.
--
-- Parameters (bound positionally by the caller):
--   $1 = path to train.csv
--   $2 = reliable-flag start date (lower bound that keeps the scan bounded)
-- Relation `target_events` (registered by the caller) has columns:
--   evkey (int), store_nbr, item_nbr, pre_start, promo_start
-- pre_start/promo_start are cast to DATE so the comparison works whether the CSV's
-- `date` column is inferred as DATE or TIMESTAMP.
SELECT
    e.evkey                       AS evkey,
    SUM(t.unit_sales)             AS pre_period_unit_sales
FROM read_csv_auto($1, header = TRUE) AS t
JOIN target_events AS e
  ON t.store_nbr = e.store_nbr
 AND t.item_nbr = e.item_nbr
WHERE t.date >= CAST($2 AS DATE)
  AND t.date >= CAST(e.pre_start AS DATE)
  AND t.date <  CAST(e.promo_start AS DATE)
GROUP BY e.evkey;
