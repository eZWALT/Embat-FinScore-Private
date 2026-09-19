# Score forecast evaluation (train companies, group-fold CV, holdout untouched)

Shipped: **reversion_quantile at horizons 1, 2, 3, 4, 5, 6**. the reversion quantile fan beat the naive fan on pinball loss with an interval above zero at horizons 1, 2, 3, 4, 5, 6

The score is not a trending series: monthly changes are negatively autocorrelated and the score is pulled toward the company's own average and the portfolio level. So extrapolating a trend adds nothing; the gain is in that pull and in the shape of the fan.

## 1. Median forecast (MAE in points, and skill = 1 - MAE / MAE(last value), 95% group-bootstrap interval)

| horizon | origins | last value | smoothed level | damped trend (Holt) | reversion model |
|---|---|---|---|---|---|
| 1 | 13878 | 3.46 | 3.65 (-5.5%; -5.9% to -5.1%) | 3.61 (-4.5%; -4.9% to -4.2%) | 3.44 (+0.4%; +0.2% to +0.6%) |
| 2 | 12674 | 5.61 | 5.71 (-1.7%; -2.0% to -1.4%) | 5.69 (-1.4%; -1.6% to -1.2%) | 5.57 (+0.6%; +0.3% to +1.0%) |
| 3 | 11542 | 7.24 | 7.27 (-0.4%; -0.7% to -0.2%) | 7.27 (-0.4%; -0.6% to -0.3%) | 7.14 (+1.4%; +0.6% to +2.0%) |
| 4 | 10536 | 8.36 | 8.35 (+0.2%; -0.0% to +0.5%) | 8.37 (-0.0%; -0.2% to +0.1%) | 8.14 (+2.7%; +1.3% to +3.9%) |
| 5 | 9591 | 9.13 | 9.08 (+0.6%; +0.3% to +0.8%) | 9.12 (+0.2%; +0.0% to +0.3%) | 8.76 (+4.1%; +2.2% to +5.7%) |
| 6 | 8683 | 9.60 | 9.53 (+0.7%; +0.4% to +1.0%) | 9.58 (+0.3%; +0.1% to +0.4%) | 9.15 (+4.7%; +2.3% to +6.7%) |

Damped trend is the tendency model. Its skill against the last value runs from -4.5% to +0.3% across horizons, against +0.4% to +4.7% for the reversion model: extrapolating a trend adds nothing, using the pull toward the company's own average does.

## 2. The fan (pinball loss over the 10/25/50/75/90% quantiles, out of fold)

| horizon | pinball naive fan | pinball reversion fan | skill (95% CI) | 50% covers (naive / reversion) | 80% covers (naive / reversion) | shipped |
|---|---|---|---|---|---|---|
| 1 | 1.412 | 1.351 | +4.4% (+3.8% to +4.9%) | 50% / 50% | 80% / 80% | reversion_quantile |
| 2 | 2.216 | 2.079 | +6.2% (+5.1% to +7.0%) | 50% / 50% | 80% / 80% | reversion_quantile |
| 3 | 2.816 | 2.577 | +8.5% (+7.0% to +9.8%) | 50% / 50% | 80% / 80% | reversion_quantile |
| 4 | 3.237 | 2.894 | +10.6% (+8.8% to +12.1%) | 50% / 50% | 80% / 80% | reversion_quantile |
| 5 | 3.519 | 3.089 | +12.2% (+10.0% to +14.0%) | 50% / 50% | 80% / 80% | reversion_quantile |
| 6 | 3.695 | 3.211 | +13.1% (+10.8% to +15.2%) | 50% / 49% | 80% / 80% | reversion_quantile |

The naive fan is the last value with pooled quantiles of the error scaled by the company's own volatility; the reversion fan is a per-horizon, per-quantile linear quantile regression on the company's features, so it is skewed (high scores can fall further than they can rise) and shifts with the deviation from the company's own average.

### Where the shipped fan is calibrated (3-month horizon, out of fold)

| group | origins | 50% covers | 80% covers |
|---|---|---|---|
| score <60 | 3800 | 50% | 80% |
| score 60-75 | 5031 | 49% | 79% |
| score 75+ | 2711 | 50% | 79% |
| history 4-7 months | 3723 | 48% | 80% |
| history 8-13 | 4590 | 50% | 80% |
| history 14+ | 3229 | 51% | 80% |
| calm (lowest third of own volatility) | 3848 | 49% | 79% |
| middle third | 3847 | 51% | 81% |
| volatile (top third) | 3847 | 50% | 79% |

### Where the shipped fan is calibrated (6-month horizon, out of fold)

| group | origins | 50% covers | 80% covers |
|---|---|---|---|
| score <60 | 2783 | 49% | 80% |
| score 60-75 | 3906 | 49% | 80% |
| score 75+ | 1994 | 49% | 79% |
| history 4-7 months | 3255 | 49% | 79% |
| history 8-13 | 4037 | 49% | 81% |
| history 14+ | 1391 | 49% | 80% |
| calm (lowest third of own volatility) | 2895 | 48% | 79% |
| middle third | 2894 | 50% | 81% |
| volatile (top third) | 2894 | 49% | 80% |

## 3. Stricter check: also hold out the later months

The group-fold CV above holds out companies but shares calendar months between train and test. Here the models are fitted only on origins whose target month is up to 2025-09 (other folds' companies) and tested on the held-out companies' origins after it. Pinball skill against the naive fan fitted the same way:

| horizon | origins | skill (95% CI) | 50% covers | 80% covers |
|---|---|---|---|---|
| 1 | 9202 | +2.2% (+0.9% to +3.5%) | 49% | 77% |
| 2 | 7998 | +2.0% (-0.4% to +4.7%) | 46% | 74% |
| 3 | 6866 | +2.3% (-2.1% to +5.7%) | 45% | 74% |
| 4 | 5860 | +4.4% (-0.1% to +8.1%) | 45% | 75% |
| 5 | 4915 | +7.0% (+3.2% to +10.5%) | 46% | 74% |
| 6 | 4007 | +8.7% (+4.6% to +12.3%) | 45% | 75% |

The gain holds out of time but is smaller than in section 2 at the short horizons, and the 80% interval covers a little under 80% on later months, so read section 2 as the optimistic end.

## 4. Seasonality

Lag-12 autocorrelation of monthly changes -0.033 (95% CI -0.067 to +0.003, 5428 pairs); lags 1-11 range -0.130 to +0.014. Calendar-month mean changes: correlation between the two years -0.42 over 9 months, with monthly means within about ±0.7 points. **No seasonal pattern found, none shipped.** 24 months give one lap and a bit: this can rule out a strong yearly pattern, not a weak one.

## 5. Do moves persist? (the tendency question)

| move over 3 months (>= 8 points) | cases | median move | next 3 months: median / mean change | at least half reversed (95% CI) | continued (>= 2 more points) |
|---|---|---|---|---|---|
| falls | 2268 | -13.2 | +0.2 / +1.7 | 28% (25% to 31%) | 35% |
| rises | 1432 | +13.0 | -1.7 / -3.4 | 31% (28% to 34%) | 28% |

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
