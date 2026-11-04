-- promoted_rows.sql
-- All promoted observations as (store_nbr, item_nbr, date), restricted to the window
-- where the onpromotion flag is reliably populated. This promoted subset is small
-- relative to the full table, so the caller detects consecutive-day runs (events) from
-- it in pandas. Streaming scan over train.csv; no full in-memory load.
--
-- Parameters:
--   $1 = path to train.csv
--   $2 = reliable-flag start date
SELECT
    store_nbr,
    item_nbr,
    date
-- Note: onpromotion is stored as the text 'True'/'False' (NULL early in the series),
-- so it is compared as a string.
FROM read_csv_auto($1, header = TRUE)
WHERE onpromotion = 'True'
  AND date >= CAST($2 AS DATE);
-- Not ordered here on purpose: ordering ~10M+ rows in SQL forces a disk-spilling sort.
-- The caller sorts the (much cheaper to hold) result in pandas during run detection.
