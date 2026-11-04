-- daily_panel.sql
-- Daily unit_sales panel for a given set of (store_nbr, item_nbr) pairs over a date
-- range. The caller registers the wanted pairs as a DuckDB relation named
-- ``target_pairs`` (columns: store_nbr, item_nbr) and binds the date bounds; the inner
-- join keeps the scan narrow so only the needed rows are materialized. Used by the
-- later panel builder (build_panel) to assemble target + donor series.
--
-- Parameters:
--   $1 = path to train.csv
--   $2 = inclusive start date
--   $3 = inclusive end date
SELECT
    t.store_nbr,
    t.item_nbr,
    t.date,
    t.unit_sales,
    (t.onpromotion = 'True') AS onpromotion  -- text 'True'/'False'/NULL -> boolean
FROM read_csv_auto($1, header = TRUE) AS t
INNER JOIN target_pairs AS p
    ON t.store_nbr = p.store_nbr
   AND t.item_nbr = p.item_nbr
WHERE t.date >= CAST($2 AS DATE)
  AND t.date <= CAST($3 AS DATE)
ORDER BY t.store_nbr, t.item_nbr, t.date;
