# Promotion_ROI_Analysis — Project Spec

> Standing reference. This is the spec the project is built from. Internal working
> document; not linked from the public README.

Project name:   Promotion_ROI_Analysis

## Goal
Measure whether real retail promotions actually made money, not just whether sales
rose during the promotion window. For each analyzed promotion: isolate its true
causal effect from normal trend and noise using interrupted time series and
synthetic control (not a naive before/after comparison), quantify cannibalization
(pulled-forward demand that reduces post-promotion sales), and compute a net profit
impact after the discount cost. Deliver an interactive, precomputed dashboard
published on GitHub Pages.

## Key questions
1. For a sample of real promotions, what is each one's TRUE effect on sales,
   isolated from the underlying trend, seasonality, and noise that would have
   occurred anyway?
2. How much of that effect is cannibalization: a post-promotion sales dip that
   offsets some or all of the gain, because customers bought ahead of their normal
   need?
3. After subtracting cannibalization and the discount cost, which promotions were
   net profitable, and which actually lost money despite looking like a "win" on a
   naive before/after view?
4. Do interrupted time series and synthetic control agree on the effect size? When
   they disagree, what does that say about how confident we should be?
5. Can a simple, explainable rule (checked a few days into a promotion) flag a
   likely net-negative promotion before it finishes, instead of only finding out
   afterward?

## Required tools
Python (pandas, NumPy, statsmodels, scikit-learn, SciPy), SQL (extraction and
aggregation), Plotly (published dashboard), Git, GitHub CLI (gh), GitHub Pages.

## Deliverables
Data pipeline, interrupted time series models, synthetic control models,
cannibalization and net-impact calculations, an early-warning rule with a backtest,
a precomputed interactive HTML dashboard published live on GitHub Pages, a clean
public README.

## Data
Corporación Favorita Grocery Sales Forecasting dataset (Kaggle): real daily unit
sales across Ecuadorian grocery stores and product families, with an `onpromotion`
flag per item per day, plus store metadata and a holiday/events calendar.
https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data
