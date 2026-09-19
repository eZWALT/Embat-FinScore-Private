# Score forecast evaluation (train companies, group-fold CV, holdout untouched)

Shipped: **reversion_quantile at horizons 1, 2, 3, 4, 5, 6**. the reversion quantile fan beat the naive fan on pinball loss with an interval above zero at horizons 1, 2, 3, 4, 5, 6

The score is not a trending series: monthly changes are negatively autocorrelated and the score is pulled toward the company's own average and the portfolio level. So extrapolating a trend adds nothing; the gain is in that pull and in the shape of the fan.

## 1. Median forecast (MAE in points, and skill = 1 - MAE / MAE(last value), 95% group-bootstrap interval)

| horizon | origins | last value | smoothed level | damped trend (Holt) | reversion model |
|---|---|---|---|---|---|
| 1 | 13878 | 3.92 | 4.14 (-5.5%; -5.9% to -5.0%) | 4.10 (-4.6%; -5.0% to -4.2%) | 3.93 (-0.1%; -0.2% to -0.1%) |
| 2 | 12674 | 6.33 | 6.43 (-1.5%; -1.8% to -1.1%) | 6.41 (-1.3%; -1.5% to -1.0%) | 6.33 (+0.1%; -0.1% to +0.4%) |
| 3 | 11542 | 8.13 | 8.14 (-0.2%; -0.4% to +0.1%) | 8.15 (-0.3%; -0.5% to -0.2%) | 8.05 (+1.0%; +0.3% to +1.6%) |
| 4 | 10536 | 9.17 | 9.14 (+0.4%; +0.1% to +0.6%) | 9.17 (+0.0%; -0.1% to +0.2%) | 8.95 (+2.5%; +1.1% to +3.7%) |
| 5 | 9591 | 9.89 | 9.82 (+0.7%; +0.4% to +1.0%) | 9.87 (+0.2%; +0.0% to +0.4%) | 9.50 (+3.9%; +2.3% to +5.5%) |
| 6 | 8683 | 10.38 | 10.27 (+1.0%; +0.7% to +1.2%) | 10.33 (+0.4%; +0.2% to +0.6%) | 9.89 (+4.7%; +2.3% to +6.9%) |

Damped trend is the tendency model. Its skill against the last value runs from -4.6% to +0.4% across horizons, against -0.1% to +4.7% for the reversion model: extrapolating a trend adds nothing, using the pull toward the company's own average does.

## 2. The fan (pinball loss over the 10/25/50/75/90% quantiles, out of fold)

| horizon | pinball naive fan | pinball reversion fan | skill (95% CI) | 50% covers (naive / reversion) | 80% covers (naive / reversion) | shipped |
|---|---|---|---|---|---|---|
| 1 | 1.704 | 1.648 | +3.3% (+2.7% to +3.8%) | 50% / 50% | 80% / 80% | reversion_quantile |
| 2 | 2.636 | 2.444 | +7.3% (+6.0% to +8.3%) | 50% / 50% | 80% / 80% | reversion_quantile |
| 3 | 3.309 | 2.929 | +11.5% (+9.9% to +13.0%) | 50% / 50% | 80% / 80% | reversion_quantile |
| 4 | 3.681 | 3.185 | +13.5% (+11.5% to +15.2%) | 50% / 50% | 80% / 80% | reversion_quantile |
| 5 | 3.929 | 3.342 | +14.9% (+12.7% to +16.7%) | 50% / 50% | 80% / 80% | reversion_quantile |
| 6 | 4.097 | 3.448 | +15.9% (+13.7% to +18.1%) | 50% / 49% | 80% / 80% | reversion_quantile |

The naive fan is the last value with pooled quantiles of the error scaled by the company's own volatility; the reversion fan is a per-horizon, per-quantile linear quantile regression on the company's features, so it is skewed (high scores can fall further than they can rise) and shifts with the deviation from the company's own average.

### Where the shipped fan is calibrated (3-month horizon, out of fold)

| group | origins | 50% covers | 80% covers |
|---|---|---|---|
| score <60 | 4101 | 50% | 79% |
| score 60-75 | 4767 | 49% | 81% |
| score 75+ | 2674 | 50% | 79% |
| history 4-7 months | 3723 | 47% | 77% |
| history 8-13 | 4590 | 50% | 80% |
| history 14+ | 3229 | 52% | 81% |
| calm (lowest third of own volatility) | 3848 | 48% | 78% |
| middle third | 3847 | 51% | 80% |
| volatile (top third) | 3847 | 50% | 81% |

### Where the shipped fan is calibrated (6-month horizon, out of fold)

| group | origins | 50% covers | 80% covers |
|---|---|---|---|
| score <60 | 2987 | 49% | 79% |
| score 60-75 | 3724 | 50% | 81% |
| score 75+ | 1972 | 49% | 79% |
| history 4-7 months | 3255 | 49% | 79% |
| history 8-13 | 4037 | 49% | 80% |
| history 14+ | 1391 | 50% | 81% |
| calm (lowest third of own volatility) | 2895 | 48% | 78% |
| middle third | 2894 | 50% | 80% |
| volatile (top third) | 2894 | 50% | 83% |

## 3. Stricter check: also hold out the later months

The group-fold CV above holds out companies but shares calendar months between train and test. Here the models are fitted only on origins whose target month is up to 2025-09 (other folds' companies) and tested on the held-out companies' origins after it. Pinball skill against the naive fan fitted the same way:

| horizon | origins | skill (95% CI) | 50% covers | 80% covers |
|---|---|---|---|---|
| 1 | 9202 | +1.5% (+0.4% to +2.5%) | 51% | 78% |
| 2 | 7998 | +4.6% (+2.7% to +6.7%) | 49% | 76% |
| 3 | 6866 | +7.8% (+5.1% to +10.5%) | 47% | 76% |
| 4 | 5860 | +6.9% (+2.6% to +10.7%) | 45% | 74% |
| 5 | 4915 | +9.5% (+6.0% to +13.0%) | 46% | 74% |
| 6 | 4007 | +11.6% (+7.8% to +14.6%) | 46% | 76% |

The gain holds out of time but is smaller than in section 2 at the short horizons, and the 80% interval covers a little under 80% on later months, so read section 2 as the optimistic end.

## 4. Seasonality

Lag-12 autocorrelation of monthly changes -0.014 (95% CI -0.051 to +0.028, 5428 pairs); lags 1-11 range -0.143 to +0.031. Calendar-month mean changes: correlation between the two years -0.21 over 9 months, with monthly means within about ±0.9 points. **No seasonal pattern found, none shipped.** 24 months give one lap and a bit: this can rule out a strong yearly pattern, not a weak one.

## 5. Do moves persist? (the tendency question)

| move over 3 months (>= 8 points) | cases | median move | next 3 months: median / mean change | at least half reversed (95% CI) | continued (>= 2 more points) |
|---|---|---|---|---|---|
| falls | 2225 | -15.1 | +0.3 / +3.9 | 28% (25% to 31%) | 25% |
| rises | 1482 | +13.7 | -1.8 / -5.1 | 32% (29% to 36%) | 29% |

Moves are heavy-tailed: the typical material fall holds (most do not recover half within 3 months) while the average one partly recovers because of a minority of large reversals; the typical rise gives some back. This is why the score alerts describe a move and the fan is asymmetric, rather than either extrapolating or assuming a return.

## 6. Per-company ARIMA (statsmodels), benchmark only

220 sampled companies, 6840 forecasts, origins with at least 8 scored months. MAE of the median forecast in points, same origins for every column.

| horizon | forecasts | last value | reversion model (shipped) | AR(1)+const | ARIMA(0,1,1) | ARIMA(1,1,1) | ARIMA(1,1,0) |
|---|---|---|---|---|---|---|---|
| 1 | 1380 | 4.01 | 4.02 | 5.11 | 4.61 | 4.81 | 4.45 |
| 2 | 1340 | 6.24 | 6.25 | 6.91 | 6.58 | 6.76 | 6.45 |
| 3 | 1160 | 8.14 | 8.11 | 8.26 | 8.21 | 8.62 | 8.42 |
| 4 | 1120 | 9.16 | 8.93 | 8.96 | 9.15 | 9.59 | 9.39 |
| 5 | 940 | 9.74 | 9.43 | 9.61 | 9.80 | 10.37 | 10.07 |
| 6 | 900 | 10.21 | 9.79 | 9.85 | 10.10 | 10.69 | 10.44 |

Per-company ARIMA from at most 22 monthly scores loses to the last value at short horizons; only the mean-reverting AR(1) with a constant gains at 4-6 months, which is the effect the pooled reversion model uses, estimated on the whole portfolio instead of one short series.

Skill = 1 - loss(candidate) / loss(baseline) on the same origins; zero is a tie. Coverage is measured on held-out folds with models fitted on the other folds. The fan says where the score usually goes from here and how far it can move; it is not a prediction of financial outcomes.
