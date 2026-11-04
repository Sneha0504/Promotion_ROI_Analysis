-- profile_train.sql
-- One-pass data-quality profile of train.csv. Single streaming scan producing the
-- headline counts used to build docs/data_quality_report.md. No full in-memory load.
--
-- Parameters:
--   $1 = path to train.csv
SELECT
    COUNT(*)                                             AS n_rows,
    MIN(date)                                            AS first_date,
    MAX(date)                                            AS last_date,
    COUNT(*) FILTER (WHERE unit_sales IS NULL)           AS n_null_sales,
    COUNT(*) FILTER (WHERE unit_sales < 0)               AS n_negative_sales,
    COUNT(*) FILTER (WHERE unit_sales = 0)               AS n_zero_sales,
    COUNT(*) FILTER (WHERE onpromotion IS NULL)          AS n_null_promo,
    COUNT(*) FILTER (WHERE onpromotion = 'True')         AS n_promo_true,
    COUNT(*) FILTER (WHERE onpromotion = 'False')        AS n_promo_false,
    MIN(date) FILTER (WHERE onpromotion IS NOT NULL)     AS first_nonnull_promo_date,
    COUNT(DISTINCT store_nbr)                            AS n_stores,
    COUNT(DISTINCT item_nbr)                             AS n_items
FROM read_csv_auto($1, header = TRUE);
