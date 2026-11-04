-- donor_scan.sql
-- One streaming pass over train.csv pulling the daily rows for every candidate donor
-- (store_nbr, item_nbr) pair across all selected events, bounded to the global analysis
-- window. The caller registers the candidate pairs as the ``donor_pairs`` relation and
-- wraps this SELECT in a ``COPY ... TO`` so the (large) result streams straight to a
-- git-ignored parquet in data/interim without being ordered (an ORDER BY here would force
-- a disk-spilling sort of tens of millions of rows). Per-event dense calendars, closure
-- masking and donor selection are then done in pandas from that parquet.
--
-- Parameters (bound positionally by the caller):
--   $1 = path to train.csv
--   $2 = inclusive global start date (earliest pre_start across events)
--   $3 = inclusive global end date (latest post_end across events)
SELECT
    t.store_nbr,
    t.item_nbr,
    t.date,
    t.unit_sales,
    (t.onpromotion = 'True') AS onpromotion  -- text 'True'/'False'/NULL -> boolean
FROM read_csv_auto($1, header = TRUE) AS t
INNER JOIN donor_pairs AS p
    ON t.store_nbr = p.store_nbr
   AND t.item_nbr = p.item_nbr
WHERE t.date >= CAST($2 AS DATE)
  AND t.date <= CAST($3 AS DATE);
