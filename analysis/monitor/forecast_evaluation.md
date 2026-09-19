# Score forecast evaluation (train companies, group-fold CV, holdout untouched)

Method shipped: **naive_last** (smoothed level did not beat the naive last value with an interval above zero, so the simpler naive last value ships). Smoothing alpha (mode over folds): 0.85. Flat forecast from the last month, fan from scaled out-of-fold errors.

| horizon (months) | origins | MAE naive | MAE smoothed | skill of smoothed vs naive (95% CI) | 50% interval covers | 80% interval covers |
|---|---|---|---|---|---|---|
| 1 | 13878 | 3.92 | 4.14 | -5.5% (-5.9% to -5.0%) | 50% | 80% |
| 2 | 12674 | 6.33 | 6.43 | -1.5% (-1.8% to -1.2%) | 50% | 80% |
| 3 | 11542 | 8.13 | 8.14 | -0.2% (-0.4% to +0.1%) | 50% | 80% |
| 4 | 10536 | 9.17 | 9.14 | +0.4% (+0.1% to +0.6%) | 50% | 80% |
| 5 | 9591 | 9.89 | 9.82 | +0.7% (+0.4% to +1.0%) | 50% | 80% |
| 6 | 8683 | 10.38 | 10.27 | +1.0% (+0.7% to +1.2%) | 50% | 80% |

Skill = 1 - MAE(smoothed) / MAE(naive); zero is a tie. Coverage is measured on held-out folds with quantiles fitted on the other folds.
The score is persistent, so the naive last value is hard to beat; the fan says how far the score usually moves, it does not say which way.
