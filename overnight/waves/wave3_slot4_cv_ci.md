# Wave 3 slot 4 — train group-fold CV intervals

Agent `67e8ef01`. Cheap LightGBM only (50 trees, `max_depth=3`). No holdout AUROC.

`holdout_power.py`: every accepted binary has <30 holdout positives. The number we quote tonight is train 5 group-fold CV mean ± fold sd.

## Files written

- `analysis/evaluate/cv_intervals.py` (owned)
- this note
- 4 rows appended to `analysis/experiments/registry.csv` (`lgbm_small_d3_n50`, `cv5_group`, metrics `auroc` + `auroc_sd`)

## Setup

- X from `data/feature_store/monthly.parquet` (22230 × 118). Y from `targets.parquet`.
- Allowed families **A+C+D+E+F+G+H**. **Never B** for both Ys. Leakage assert passed; no `b_` in X.
- Lags 1 and 3 on family X (not meta). `n_x=278` after lags (`group_size`, `n_banking` kept).
- 5 group-folds, seed `20260918`. Holdout companies excluded from every fit and every val fold. Never scored.
- Reused `gbm_y3y6` allowed-col / lag / leakage helpers. Did not refit the 400-tree model.

## Quote (train CV only)

| y | mean AUROC | fold sd | folds (0..4) | train n / pos / rate | coverage |
|---|-----------:|--------:|---|---|---:|
| `y3_recover_cash_6m` | **0.762** | **0.016** | 0.773, 0.739, 0.765, 0.779, 0.755 | 5648 / 402 / 7.12% | 0.267 |
| `y2_neg_2of3` | **0.540** | **0.117** | 0.344, 0.629, 0.622, 0.531, 0.573 | 17356 / 1271 / 7.32% | 0.820 |

Exact: y3 **0.7622 ± 0.0158**; y2 **0.5400 ± 0.1166**.

## What this means

- **Y3 is the quoteable claim.** Small tree is stable across groups (sd 0.016). Slightly above the earlier 400-tree CV 0.71. Stressed-only rows (label already NaN off that path).
- **Y2 is chance and unstable.** Mean 0.54, sd 0.12. Fold 0 inverts (0.344, 374 val positives). Matches the parked `lightgbm_panel` CV ~0.54. Do not promote.

## What failed

- Y2 group-fold signal does not survive a depth-3 stump ensemble. High fold variance; one fold worse than chance.
- Holdout AUROC is not computable as a claim (LOW_POWER). Not computed here.

## Next idea

- Quote Y3 0.762 ± 0.016 as the night CV number. Park Y2 GBM.
- If anyone wants a tighter Y3 interval, bootstrap the same 5-fold scores — do not add holdout AUROC.
- Y2 needs a different X story (or stay with Javier’s B-using score, which we cannot use as a fair model).
