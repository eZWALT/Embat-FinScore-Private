# Y1 rest — inflow h1/h3 and liquidity h3

Agent `a41f9c72`. Holdout 72 never in a fit. Win rule: holdout median-norm MAE vs hist_mean. Quote CV. Variant KEEP only if holdout gap vs hist ≥ 0.02.

X: families A–H, lags 1 and 3, meta `group_size` + `n_banking`. n_x=308 (302 after log-size drop).

## Quote table (median-norm MAE)

| series | model | CV MAE | CV OOF med-norm (gbm / hist / last) | holdout med-norm (gbm / hist / last) | holdout MAE € | cos beat vs hist / last | verdict |
|--------|--------|--------:|-------------------------------------|--------------------------------------|---------------|-------------------------|---------|
| y1_in_h1 | lightgbm_y1 | 3.78m | **0.726** / 0.781 / 0.822 | 0.799 / **0.795** / 0.981 | 3.02m vs hist 2.90m | 0.54 / 0.77 | **PARK** |
| y1_in_h3 | lightgbm_y1 | 4.22m | 0.816 / 0.821 / 0.893 | **0.817** / 0.856 / 0.878 | 3.30m vs hist 3.18m | 0.51 / 0.76 | **KEEP thin** |
| y1_liq_h3 | lightgbm_y1 | 4.61m | 0.632 / 0.645 / **0.598** | 0.568 / 0.625 / 0.586 | 7.87m vs last 13.27m | 0.51 / 0.64 | **PARK path** |
| y1_in_h1 | last_resid | 2.95m | 0.738 / 0.781 / 0.822 | 0.798 / **0.795** / 0.981 | 3.41m | — | **PARK** |
| y1_in_h1 | log_size | 3.77m | 0.733 / 0.781 / 0.822 | 0.786 / 0.795 / 0.981 | 2.98m | — | **PARK** (gap 0.009 < 0.02) |
| y1_in_h1 | hist_resid | 3.70m | 0.777 / 0.781 / 0.822 | 0.763 / 0.795 / 0.981 | 2.91m vs hist 2.90m | 0.44 / 0.76 | **PARK** (holdout gap 0.033 but CV 0.005; beat-share 44%) |

Holdout company beat-share vs hist is a coin flip (~51–54%). The h3 median-norm WIN is not a majority of firms.

## Persistence (train only, no model)

| series | h | n | Pearson | Spearman |
|--------|--:|--:|--------:|---------:|
| op_in | 1 | 19943 | 0.813 | 0.765 |
| op_in | 3 | 17515 | 0.732 | 0.726 |
| liq | 1 | 19746 | 0.882 | 0.909 |
| liq | 3 | 17356 | 0.708 | 0.848 |

Inflow is mean-reverting enough that hist_mean beats last-value. Liquidity levels persist: last-value is the honest Q1 snapshot.

Calendar month-of-year after company-demeaning is empty (train R² 0.0003 for `op_in`, 0.0008 for `liq`). A seasonal-naive residual was not the right Pass 2 try; `d0748ae4` seasonal_naive also only covers ~28% of holdout rows.
