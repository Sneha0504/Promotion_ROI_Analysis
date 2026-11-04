-- candidate_series.sql
-- Per (store_nbr, item_nbr): promoted-day count, total observed days, date span, and
-- total unit_sales. Used to shortlist item-stores that have enough promotion activity
-- and history to be worth scanning for events. Runs as a streaming scan over the
-- 4.7 GB train.csv; the file is never loaded into memory.
--
-- Parameters (bound positionally by the caller):
--   $1 = path to train.csv
--   $2 = reliable-flag start date (promo flag is null early in the series)
SELECT
    store_nbr,
    item_nbr,
    COUNT(*)                                        AS n_days,
    COUNT(*) FILTER (WHERE onpromotion = 'True')    AS n_promo_days,
    SUM(CASE WHEN unit_sales > 0 THEN 1 ELSE 0 END) AS n_nonzero_days,
    MIN(date)                                       AS first_date,
    MAX(date)                                       AS last_date,
    SUM(unit_sales)                                 AS total_unit_sales
FROM read_csv_auto($1, header = TRUE)
WHERE date >= CAST($2 AS DATE)
GROUP BY store_nbr, item_nbr
HAVING COUNT(*) FILTER (WHERE onpromotion = 'True') > 0;
