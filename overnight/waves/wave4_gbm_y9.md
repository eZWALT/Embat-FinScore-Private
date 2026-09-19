# Wave 4 — LightGBM Y9 fee/interest pressure

- **When:** 2026-09-19 ~00:23 CEST
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3` (lightgbm)
- **Holdout:** `analysis/splits/holdout_companies.csv` (72 companies). Never fitted.
- **X:** `data/feature_store/monthly.parquet` (22230 × 118) → 88 allowed base cols, 260 after lags 1/3
- **Y:** `data/feature_store/targets.parquet` (both Y9 columns present; `y9_fees.build` not needed)
- **Re-run:** `python -m analysis.models.gbm_y9`

Y9 is brief questions **3 (turning)** and **5 (why)**: sustained fee+interest / inflow pressure (FinRegLab NSF/fee). Not a bankruptcy label. Not a 0–100 score.

## Files written

- `analysis/models/gbm_y9.py`
- `overnight/waves/wave4_gbm_y9.md` (this file)
- 10 rows appended to `analysis/experiments/registry.csv` (agent `634720f6`, R4 / wave 4)

Did not edit `gbm_y3y6.py`, `y9_fees.py`, or `product/`.

## Setup

5 group-fold CV on train groups (`FOLD_SEED=20260918`), then one fit on all train and one holdout pass. Lags 1 and 3 on family X (not meta). Dummy = constant 0.5. Best single = train-only sign, ≥25% labeled-train coverage (min 200), scored OOF on the same folds.

**X:** families **A B C D E G H**. **Never F.** Also dropped every column starting with `a_fin_cost` or `a_fc` (including lags). `protocol.leakage_check` turns `a_fin_cost` into `a_fin_cost_` and **misses the base column**; the model drops it itself. Store drop: `a_fin_cost` only (no `a_fc*` in the parquet).

KEEP only if CV beats dummy **and** the best single by a clear margin (`> 0.02`). Otherwise PARK. Quote CV; holdout AUROC is secondary.

## CV AUROC (the quote)

| Y | train n / pos / rate | cov | **CV AUROC** | dummy | best single (OOF) | gap vs single | hold n / **n_pos** / AUROC | **verdict** |
|---|---------------------:|----:|-------------:|------:|-------------------|--------------:|---------------------------:|-------------|
| y9_fee_r_ownp80 (primary) | 9,591 / 1,350 / **14.08%** | 43.1% | **0.554** | 0.500 | `a_out6` (+) **0.565** | **−0.011** | 392 / **73** / 0.625 | **PARK** |
| y9_fee_spike | 7,879 / 1,502 / **19.06%** | 35.4% | **0.576** | 0.500 | `a_op_in` (−) **0.563** | **+0.013** | 292 / **95** / 0.604 | **PARK** |

Holdout n_pos is 73 / 95 (above the 30-event floor). Still quote CV: fold AUROC is noisy and early stopping collapsed to 1–3 trees (same pattern as Y2 / Y8).

## Primary — y9_fee_r_ownp80

CV **0.554** beats dummy (gap 0.054) and **loses** to `a_out6` (0.565). Next singles (train): `c_n_days_with_tx` 0.567, `a_growth_3` 0.567 (−), `a_n_tx` 0.565. Gain is inflow/growth/activity (`a_in6_lag3`, `a_growth_3`, `a_in3_lag3`, `a_uncat_share`) — size/ops persistence, not a fee-pressure story once F and `a_fin_cost` are gone. Folds: 0.567 / 0.587 / 0.514 / 0.548 / 0.557 (sd 0.027).

## Secondary — y9_fee_spike

CV **0.576** beats dummy (gap 0.076) and only **+0.013** over `a_op_in` (−). Below the 0.02 KEEP bar. Top gain is the same univariate (`a_op_in`, `a_growth_3`). Inverse size / recent inflow drop is the whole signal. Folds: 0.506 / 0.588 / 0.581 / 0.613 / 0.591 (sd 0.041).

## What failed

- **No clear multivariate lift** after the correct leak screen (never F, never `a_fin_cost` / `a_fc`). Early stopping hit 1–3 trees on every fold; final `n_trees` floored at 50.
- Primary GBM is **worse** than trailing-6m outflow. Spike GBM is a rounding-level edge over `a_op_in`.
- Did not write a 0–100 score. Did not touch `product/`. Did not overwrite `gbm_y3y6.py`.

## Next idea

Park both Y9 GBMs. The labels stay accepted (turning / why). A later pass should treat `a_out6` / `a_op_in` as the baselines to beat, or split fee vs `interest_charge` as two numerators (wave-3 note) — still own-history, still no F / `a_fin_cost`. Do not retune trees on this bake-off.
