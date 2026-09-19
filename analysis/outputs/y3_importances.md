# Y3 cash-recovery importances (`y3_recover_cash_6m`)

Generated `2026-09-19T00:16` by `analysis/models/explain_y3.py`.
Refit of the accepted stressed-only LightGBM on **train groups only**.
Holdout is not used to fit, stop, SHAP, or permute.

## Setup

- Y: `y3_recover_cash_6m` (1 if stressed at t and some 3 consecutive months in t+1..t+6 have runway ≥ 3).
- X: families **A+C+D+E+F+G+H**, never B. Lags [1, 3]. Meta: `group_size`, `n_banking`.
- Train labeled (stressed): **5648** rows, **402** positives, rate **0.0712** (5648/21157 = 26.7% of train company-months).
- Group-fold CV AUROC: **0.710** (trees used in final fit: 50). Published bake-off claim is CV **0.710**; holdout 0.899 / 14 events is not the claim.
- Importance source: **mean |TreeSHAP| (shap 0.52.0 TreeExplainer)**.
- Sign `+` = higher feature value → higher P(recover). Primary sign is Spearman(feature, TreeSHAP); univariate AUROC sign is a check.
- Size flag: `|ρ| ≥ 0.85` vs `log1p(a_in3)` or `log1p(|a_op_in|)` on train stressed rows (NEAR_SIZE at 0.7). Runway copy: `|ρ| ≥ 0.8` vs `b_runway` (NEAR at 0.5).

## Top 15

| rank | feature | sign | mean\|SHAP\| | gain | perm ΔAUROC | ρ_size | ρ_runway | ρ_liq | univ AUROC | flags |
|-----:|---------|:----:|------------:|-----:|------------:|-------:|---------:|------:|-----------:|-------|
| 1 | `c_ss_month` | − | 0.276 | 9667.8 | 0.034 | 0.332 | 0.176 | 0.251 | − 0.690 | — |
| 2 | `c_salary_month` | − | 0.187 | 2327.2 | 0.020 | 0.315 | 0.104 | 0.179 | − 0.672 | — |
| 3 | `a_transfer` | − | 0.128 | 1390.8 | -0.001 | 0.010 | -0.097 | 0.011 | − 0.572 | — |
| 4 | `a_n_tx` | − | 0.125 | 1769.3 | 0.030 | 0.651 | -0.024 | 0.226 | − 0.714 | SIZE_STEM |
| 5 | `a_transfer_lag1` | − | 0.113 | 1277.3 | 0.002 | 0.020 | -0.081 | 0.009 | − 0.562 | — |
| 6 | `a_op_in` | − | 0.107 | 1094.5 | 0.020 | 0.998 | 0.058 | 0.395 | − 0.685 | SIZE_STEM,SIZE |
| 7 | `e_dso_proxy` | − | 0.102 | 1081.2 | 0.005 | -0.014 | -0.037 | -0.017 | + 0.552 | — |
| 8 | `c_n_days_with_tx` | − | 0.095 | 1029.9 | 0.001 | 0.572 | 0.003 | 0.212 | − 0.723 | SIZE_STEM |
| 9 | `f_ds_r_lag1` | − | 0.093 | 1993.6 | -0.000 | 0.256 | -0.052 | 0.077 | − 0.623 | — |
| 10 | `h_n_siblings_active_lag3` | − | 0.089 | 1415.7 | 0.000 | -0.043 | -0.113 | -0.098 | − 0.573 | — |
| 11 | `a_in3_lag3` | − | 0.084 | 1527.4 | 0.001 | 0.812 | 0.062 | 0.458 | − 0.638 | SIZE_STEM,NEAR_SIZE |
| 12 | `a_transfer_lag3` | − | 0.075 | 920.3 | 0.000 | 0.022 | -0.068 | -0.006 | − 0.546 | — |
| 13 | `h_group_size_lag3` | − | 0.071 | 2172.4 | 0.005 | -0.040 | -0.106 | -0.090 | − 0.584 | — |
| 14 | `a_net_margin` | − | 0.062 | 1361.0 | 0.001 | 0.279 | -0.008 | -0.049 | − 0.553 | — |
| 15 | `f_ds_r` | − | 0.060 | 842.5 | 0.007 | 0.257 | -0.045 | 0.082 | − 0.625 | — |

### What each of the top 10 is

1. `c_ss_month` (−) — any social-security booking this month.
2. `c_salary_month` (−) — any salary booking this month.
3. `a_transfer` (−) — signed net transfers this month.
4. `a_n_tx` (−) — transaction count this month.
5. `a_transfer_lag1` (−) — signed net transfers this month (t−1).
6. `a_op_in` (−) — operational inflow this month.
7. `e_dso_proxy` (−) — AR open / this-period AR issued.
8. `c_n_days_with_tx` (−) — distinct booking days this month.
9. `f_ds_r_lag1` (−) — trailing debt-service / inflow (t−1).
10. `h_n_siblings_active_lag3` (−) — other group companies with txs this month (t−3).

### Notes on signs and stability

- Sign is the model direction (Spearman of the feature with TreeSHAP). Univariate AUROC is a check.
- Sign disagreement (report SHAP): `e_dso_proxy` SHAP − vs univ + 0.552.
- High |SHAP| but fold-permutation ΔAUROC ≈ 0 (unstable rank): `a_transfer`, `a_transfer_lag1`, `c_n_days_with_tx`, `f_ds_r_lag1`, `h_n_siblings_active_lag3`.

## Size proxy and runway leak

Y3 vs `log1p(a_in3)` AUROC on train stressed = 0.380 (acceptance gate is < 0.60; label is not a size clone).
A/C columns in X: 96 (lags included). Worst |ρ| vs `b_runway` on train stressed: `a_growth_12_lag3` ρ=0.199.
No A/C feature reaches |ρ|≥0.80 vs `b_runway` or `b_liq`. The model cannot reconstruct `liq / (out3/3)` from A/C: A has the burn (`a_out3`) but not the cash stock. Using `a_out3` would be a burn signal, not a leaked runway.
In the top 15: SIZE clones `a_op_in`; no runway/liq copy.
Highest A/C |ρ| vs runway (diagnostic, not X-ranked):

| feature | ρ_runway | ρ_liq | ρ_size |
|---|---:|---:|---:|
| `a_growth_12_lag3` | 0.199 | 0.186 | 0.274 |
| `c_ss_month_lag1` | 0.183 | 0.256 | 0.327 |
| `c_ss_month` | 0.176 | 0.251 | 0.332 |
| `c_ss_month_lag3` | 0.169 | 0.238 | 0.309 |
| `a_uncat_share` | -0.136 | -0.118 | 0.012 |

## Judge paragraph — what would move recovery odds

Among already-stressed company-months (reconstructed liq < 0 or runway < 1 at t), 6-month cash recovery (a 3-month stretch with runway ≥ 3) is more likely when the month is operationally quiet: no social-security or salary booking (`c_ss_month`, `c_salary_month` −), fewer transactions and booking days (`a_n_tx`, `c_n_days_with_tx` −), and smaller signed transfers (`a_transfer` −). Odds fall when payroll-like outflows continue, when last month’s debt service already ate inflow (`f_ds_r_lag1` −), and when receivables sit long relative to new billings (`e_dso_proxy` − in the model; the univariate sign is a weak flip, so do not lean on DSO alone). That is a de-escalation / mean-reversion story among quiet stressed firms, not “bigger firms bounce back.” `a_op_in` is a euro-level size clone (|ρ|≥0.85 vs log inflow) and is not a lever to quote. Among the stressed its sign is negative: a high-inflow month is a busy stressed month, not a healthy rebound. Family B is not in X. No top feature copies current runway (|ρ| vs `b_runway` < 0.80; worst A/C is `a_growth_12_lag3` at 0.20). A has burn (`a_out3`) but not the cash stock, so it cannot rebuild `liq/(out3/3)`. Quote CV AUROC 0.71 vs a dummy 0.50; holdout 0.90 is 14 events.

## Top 10 with signs (return value)

1. `c_ss_month` −
2. `c_salary_month` −
3. `a_transfer` −
4. `a_n_tx` −
5. `a_transfer_lag1` −
6. `a_op_in` −
7. `e_dso_proxy` −
8. `c_n_days_with_tx` −
9. `f_ds_r_lag1` −
10. `h_n_siblings_active_lag3` −

## Re-run

```bash
/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.explain_y3
```
