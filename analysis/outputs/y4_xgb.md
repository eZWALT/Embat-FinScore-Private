# Y4 XGBoost — `y4_ds_r_double`

- **When:** 2026-09-19 00:52–01:22 CEST
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.models.xgb_y4 --variant xgb_d3_n50`
- **Holdout:** 72 companies, seed 20260918. Never fitted.
- **Y:** `y4_ds_r_double` only (train 2,370 / 329 / **13.88%**). Other Y4 columns still rejected in `y_acceptance.csv` — not revived.
- **X:** A (no `a_debt*` / `a_fin_cost*`) + B C D E G H. **Never F.** Lags 1, 3.
- **Quote:** train group-fold CV (5 folds). Holdout is LOW_POWER (`n_pos=16`).
- **Brief:** Q3 turning / Q5 why (debt-service pressure doubles). Not bankruptcy. Not a 0–100.

## Decision

**PARK** the tree models. Same honesty as Y5 XGB: they do not beat a single allowed feature, and holdout inverts.

The KEEP bar is the best *fixed* OOF single on the same folds: **`d_cust_hhi_lag3` 0.605**. A noisy per-fold train-pick looked like 0.508 and briefly produced a false KEEP; that pick is not the bar.

## Table (train group-fold CV)

| variant | CV AUROC ± sd | dummy | best single (OOF) | gap vs single | n_x | trees | collapsed | hold n_pos / AUROC | decision |
|---|---:|---:|---|---:|---:|---:|---|---|---|
| xgb_es (400, depth 4, ES 40) | 0.585 ± 0.039 | 0.50 | `d_cust_hhi_lag3` 0.605 | −0.020 | 241 | 41 | **yes** (fold 3 = 7) | 16 / 0.467 | **PARK** |
| xgb_d3_n50 | 0.561 ± 0.050 | 0.50 | 0.605 | −0.044 | 241 | 50 | no | 16 / 0.467 | **PARK** |
| lgb_d3_n50 | 0.540 ± 0.043 | 0.50 | 0.605 | −0.065 | 241 | 50 | no | 16 / 0.479 | **PARK** |
| xgb_d3_n50_nosize | 0.551 ± 0.048 | 0.50 | 0.605 | −0.054 | 199 | 50 | no | 16 / 0.455 | **PARK** |
| lgb_d3_n50_nosize | 0.544 ± 0.041 | 0.50 | 0.605 | −0.061 | 190 | 50 | no | 16 / 0.481 | **PARK** |
| xgb_d3_n50_ratios | 0.509 ± 0.063 | 0.50 | 0.605 | −0.095 | 146 | 50 | no | 16 / 0.503 | **PARK** |
| xgb_d3_n50_d (family D only) | 0.492 ± 0.039 | 0.50 | 0.605 | −0.113 | 27 | 50 | no | 16 / 0.501 | **PARK** |
| xgb_d3_n50_d2 (HHI+supp stems) | 0.515 ± 0.028 | 0.50 | 0.605 | −0.090 | 6 | 50 | no | 16 / 0.466 | **PARK** |
| d_zavg (not a tree) | 0.634 ± 0.076 | 0.50 | 0.605 | +0.029 | 2 | 0 | no | 16 / 0.368 | **PARK** (next idea, not XGB) |

Complete-case on rows where `d_cust_hhi_lag3` is defined: XGB d3_n50 **0.581** vs single **0.605** (mean n_va ≈ 170). Still loses.

## Screens

- Leak vs `f_ds_r` / lags on train labeled: max \|ρ\| = **0.411** (`c_n_days_with_tx`). Fail ≥ 0.80. **Pass.**
- `d_cust_hhi_lag3` vs `f_ds_r`: ρ = **0.234** (n = 848). Not a ds_r clone.
- Size `log1p(|a_in3|)` AUROC = **0.493** (need < 0.60). `log1p(|a_op_in|)` = **0.443** (matches acceptance). **Pass.**
- Dropped clones: `a_fin_cost*`, `a_debt*` (ds_r numerator — would copy the label), all `f_*`. CAT_MAP `m_debt*` shares are not in `monthly.parquet` (catmix not merged).

## Why the trees lose

SHAP names on the train-only d3_n50 fit (no holdout, no plots): `a_growth_3`, `group_size`, `a_growth_3_lag1`, `a_n_tx`, `a_growth_12_lag3`, `a_out3`, `h_share_group_in_lag3`, `c_n_tx`, `d_n_cust`, `a_n_tx_lag1`.

Gain leaders are activity / group scale, not debt service. The model is not reading ds_r pressure. It is also not using the winning univariate (`d_cust_hhi_lag3` is only 3rd on gain).

Family-D-only XGB (27 cols) falls **below dummy**. Two-stem XGB on HHI + suppliers (6 cols, `--variant xgb_d3_n50_d2`) is CV **0.515** — loses to the z-average (0.634) and to the single (0.605). Trees wash out the signal.

## Honest single (the thing that works)

`d_cust_hhi_lag3` (+): train coverage 35.8% (848 / 2,370), 116 positives, train AUROC 0.592, OOF 0.605. Positives have higher customer HHI (0.64 vs 0.54). Missingness is not the signal (is_notna AUROC 0.497).

Neighbours: `d_cust_top1_lag3` 0.604, `d_n_supp_lag3` (−) 0.600. `d_cust_hhi_lag3` vs `d_cust_top1_lag3` ρ = **0.979** (same stem — do not stack). vs `d_n_supp_lag3` ρ = −0.29. Best non-D runner-up is `e_delay_coll` 0.588 (collections delay) — still below HHI. Concentrated customers / fewer suppliers three months earlier — Q5 why, Q6 lead. Not inflow size.

OOF lead time (same 5 folds, sign from train fold):

| stem | t | t−1 | t−3 |
|---|---:|---:|---:|
| `d_cust_hhi` (+) | 0.583 | 0.603 | **0.605** |
| `d_cust_top1` (+) | 0.583 | 0.602 | **0.604** |
| `d_n_supp` (−) | 0.558 | 0.576 | **0.600** |

Lag 3 is best for all three. The turn is visible a quarter earlier.

Train labeled complete-case Y rate by `d_cust_hhi_lag3` quintile (n = 848): 8.2% / 15.4% / 10.6% / 11.8% / **22.4%** (HHI > 0.975). Near-monopoly customers, then inflow crash.

HHI is sticky (train acf1 = 0.95, acf3 = 0.86) but not only a company type: company-median HHI scores month-level Y at 0.545, contemporaneous HHI 0.574, lag3 **0.592**. ΔHHI over 3 months is a coin flip (AUROC 0.51). The **level** at t−3 matters, not a sudden concentration spike.

A train-only signed z-average of `d_cust_hhi_lag3` (+) and `d_n_supp_lag3` (−) posts OOF **0.634 ± 0.076** (gap +0.029 vs the single). Reproducible via `--variant d_zavg`. Fold 2 is 0.518; holdout is **0.368** because D coverage on holdout labeled rows is 21 / 2 pos. Next idea, not an XGB KEEP, and not holdout-ready.

## Mapping

Y4 doubling of debt-service / inflow is *who is turning* (Q3) and *why* (Q5). On train labeled rows the doubling is mostly a **denominator crash**: positives have median `in3[t+3]/in3[t] = 0.36` and median `ds3[t+3]/ds3[t] = 1.15`. 80% of positives drop inflow >20%; 49% raise debt service >20%. `d_cust_hhi_lag3` is higher among the inflow-drop positives (0.65 vs 0.51). So the honest why is concentrated customers, then inflow falls, then `ds_r` doubles — Q6 visible a quarter earlier. Trees that grab `a_growth_*` / `a_n_tx` are late to that story and still lose to HHI.

The label stays accepted. Do not write a 0–100 from this bake-off.
