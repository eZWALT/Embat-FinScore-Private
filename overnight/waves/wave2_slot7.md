# Wave 2 · slot 7 — Y3 recovery + Y6 activity

- **Files:** `analysis/targets/y3_recovery.py`, `analysis/targets/y6_activity.py`
- **Grid:** monthly `company_id × period` (22,230 rows, 1,286 companies)
- **Train only** (holdout `analysis/splits/holdout_companies.csv` excluded from rates/AUROC): 21,157 company-months, 1,214 companies
- **Thresholds:** catalogue-fixed. Not searched on train or holdout. No module `REF`.
- **Liq / runway:** `cash_month_panel` from `y1_forecast` (same unwind as Y2).
- **Re-run:** `python -m analysis.targets.y3_recovery` and `python -m analysis.targets.y6_activity`
- **Smoke:** `/tmp/wave2_slot7_smoke.py` (not in the repo)

Y1/Y2 were not edited. Accepted binary from wave 1 remains `y2_neg_2of3` only.

## Y3 Recovery (from stressed at t)

Stressed = `liq < 0` OR `runway < 1` at t. Label is NaN if not stressed or t+6 is past the sample or any of t+1..t+6 has missing runway.

Recover = some 3 consecutive months in t+1..t+6 with `runway >= 3`.  
`y3_recover_6m` also requires open-AP overdue>30d share ≤ **0.50** on each of those 3 months (catalogue “not high” = not a majority of open AP; no open AP counts as not high). AP stock matches family E `e_ap_overdue_30` (max abs diff ~0).

| META | value |
|------|--------|
| `forbidden_x_families` | `["b"]` |
| `y3_recover_6m` extra | also `["e"]` — overdue is in the label; do not use E for that variant |
| `y3_recover_cash_6m` | `["b"]` only — X may use E |

Relevant rate is **among stressed** (5–50%). Size = AUROC of `log1p(|op_in|)` vs Y (same formula as Y2). Contract accepts raw AUROC < 0.60.

| column | train stressed n | pos | rate stressed | rate all rows | size AUROC | two-sided | accepted |
|--------|-----------------:|----:|--------------:|--------------:|-----------:|----------:|----------|
| y3_recover_6m | 5,648 | 241 | **4.27%** | 1.14% | 0.329 | 0.671 | **REJECTED** |
| y3_recover_cash_6m | 5,648 | 402 | **7.12%** | 1.90% | 0.315 | 0.685 | **ACCEPTED** |

Stressed-conditional coverage: 5,648 / 21,157 = 26.7% of train rows (725 companies); 174 companies have at least one cash-recover positive.

## Y6 Activity / going-concern

Salary / n_tx / last_tx recomputed from `transactions` (not imported from `ops.py`).  
`META.forbidden_x_families = ["a", "c"]` (same window). Horizon 3.

| column | definition |
|--------|------------|
| `y6_zero_in_3` | t has activity (`in3 > 0` or `n_tx > 0`) and `op_in == 0` in t+1, t+2 and t+3 |
| `y6_missed_payroll` | usual payroll (≥3 of last 6 months ending at t have `category=salary`) and t+1..t+3 all have no salary |
| `y6_silent_60` | last tx as of t+3 month-end is >60 days before that end, after prior activity at t |

Acceptance: train labeled base rate 5–30% and size AUROC < 0.60 (CONTRACT).

| column | train n | pos | base rate | size AUROC | two-sided | accepted |
|--------|--------:|----:|----------:|-----------:|----------:|----------|
| y6_zero_in_3 | 17,083 | 1,490 | **8.72%** | 0.121 | 0.879 | **ACCEPTED** |
| y6_missed_payroll | 6,004 | 367 | **6.11%** | 0.401 | 0.599 | **ACCEPTED** |
| y6_silent_60 | 17,515 | 525 | **3.00%** | 0.180 | 0.820 | **REJECTED** |

`y6_missed_payroll` labeled only on usual-payroll rows (579 train companies). `y6_zero_in_3` / `y6_silent_60` cover ~81–83% of train rows (horizon tail + activity gate).

## What failed

- **y3_recover_6m:** adding “AP overdue>30d not high” (even at a majority cut 0.50, treating missing AP as not high) drops the stressed-conditional rate from 7.12% to 4.27% (under 5%). Cash recovery and clean AP rarely coincide. Strict “no AP >30d” (share = 0) is 3.4%.
- **y6_silent_60:** 60-day silence by t+3 is rare (3.00%). Contemporaneous recency>60 is 2.57%. The 60-day cut was not relaxed.
- **Inverse size (not used to reject):** contract/Y2 raw AUROC < 0.60. Y5’s two-sided `max(auc, 1-auc) ≥ 0.60` would also reject `y3_recover_cash_6m` (0.685) and `y6_zero_in_3` (0.879: already-small inflow predicts a zero-inflow spell). `y6_missed_payroll` still clears two-sided (0.599).

## Next idea (later wave; do not retune here)

Use `y3_recover_cash_6m` as the R2 recovery binary (X may include E). Keep `y3_recover_6m` as a cross-source variant only if a new column (e.g. AP condition on the last month of the window only) is added — do not fit the 0.50 cut. For silence, a 30-day recency event or “onset silent after recency≤60 at t” needs a new column; do not change `y6_silent_60`. If the bake-off adopts Y5 two-sided size, treat `y6_zero_in_3` as a size/activity proxy and prefer `y6_missed_payroll`.
