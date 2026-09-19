# Wave 3 · slot 7 — explain Y3 cash-recovery LightGBM

- **Files:** `analysis/models/explain_y3.py`, `analysis/outputs/y3_importances.md` (this note). Did **not** edit `gbm_y3y6.py`, Y modules, or the store.
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3` (installed `shap==0.52.0`; TreeExplainer ran).
- **Re-run:** `python -m analysis.models.explain_y3`
- **Holdout:** never used to fit, stop, SHAP, or permute. Family B never in X (only a post-hoc |ρ| screen vs `b_runway` / `b_liq`).

## Setup

Same spec as the accepted bake-off: `y3_recover_cash_6m`, stressed-only labeled rows, X = A+C+D+E+F+G+H, lags 1 and 3, meta `group_size` + `n_banking`, 5 group-fold CV then one train fit (`n_trees=50`). CV AUROC **0.710** (matches the claim). Train labeled **5,648** / 21,157 company-months (**26.7%**), 402 positives, rate 7.12%. 278 X columns.

Importance = mean |TreeSHAP| on the train fit. Sign = Spearman(feature, SHAP): **−** = higher feature → lower P(recover). Fold-permutation ΔAUROC is a train-fold check only.

## Top 10 with signs

| rank | feature | sign | note |
|-----:|---------|:----:|------|
| 1 | `c_ss_month` | − | any social-security booking this month |
| 2 | `c_salary_month` | − | any salary booking this month |
| 3 | `a_transfer` | − | signed net transfers; high \|SHAP\|, perm ≈ 0 (unstable) |
| 4 | `a_n_tx` | − | tx count (activity stem, ρ_size 0.65) |
| 5 | `a_transfer_lag1` | − | transfers t−1; perm ≈ 0 |
| 6 | `a_op_in` | − | **SIZE** clone (ρ 0.998 vs log inflow) |
| 7 | `e_dso_proxy` | − | SHAP −; univariate flips to + 0.55 |
| 8 | `c_n_days_with_tx` | − | booking days; best single in the bake-off |
| 9 | `f_ds_r_lag1` | − | debt-service / inflow at t−1 |
| 10 | `h_n_siblings_active_lag3` | − | active siblings t−3; perm ≈ 0 |

Ranks 11–15: `a_in3_lag3` (−, NEAR_SIZE ρ 0.81), `a_transfer_lag3` (−), `h_group_size_lag3` (−), `a_net_margin` (−), `f_ds_r` (−). Permutation agrees on the stable core: `c_ss_month` ΔAUROC 0.034, `a_n_tx` 0.030, `c_salary_month` / `a_op_in` 0.020.

## Size proxy / runway leak

- Label vs `log1p(a_in3)` AUROC on train stressed = **0.380** (gate < 0.60; not a size Y).
- Only hard size clone in the top 15: **`a_op_in`**. `a_n_tx` / `c_n_days_with_tx` are activity stems, not |ρ|≥0.85. `a_in3_lag3` is NEAR_SIZE (0.81).
- **No A/C near-copy of runway.** Worst |ρ| vs `b_runway` among 96 A/C columns (lags in): `a_growth_12_lag3` **0.199**. No |ρ|≥0.50 vs runway. A has burn (`a_out3`) but not cash stock, so it cannot rebuild `liq/(out3/3)`. `a_out3` is not in the top 15.

## Judge paragraph — what would move recovery odds

Among already-stressed company-months (reconstructed liq < 0 or runway < 1 at t), 6-month cash recovery (a 3-month stretch with runway ≥ 3) is more likely when the month is operationally quiet: no social-security or salary booking (`c_ss_month`, `c_salary_month` −), fewer transactions and booking days (`a_n_tx`, `c_n_days_with_tx` −), and smaller signed transfers (`a_transfer` −). Odds fall when payroll-like outflows continue, when last month’s debt service already ate inflow (`f_ds_r_lag1` −), and when receivables sit long relative to new billings (`e_dso_proxy` − in the model; the univariate sign is a weak flip, so do not lean on DSO alone). That is a de-escalation / mean-reversion story among quiet stressed firms, not “bigger firms bounce back.” `a_op_in` is a euro-level size clone and is not a lever to quote (sign is negative among the stressed). Family B is not in X; no top feature copies current runway. Quote CV AUROC 0.71 vs a dummy 0.50; holdout 0.90 is 14 events.

## What failed

- Nothing blocking. SHAP imported and ran.
- `a_transfer` / sibling / DSO ranks are SHAP-heavy and permutation-light — do not brief those as durable levers.
- `h_group_size_lag3` is a lag of a static column (same as contemporaneous). Not a leak; just wasted rank.

## Next idea

Drop `a_op_in` / other euro levels and static `h_group_size` / `group_size`, keep payroll flags + `c_n_days_with_tx` + `f_ds_r`. A 4–6 feature model should be the next thing to beat on CV (the bake-off already said GBM’s lift over `c_n_days_with_tx` is small).
