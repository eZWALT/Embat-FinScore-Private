# SHAP — y3_recover_cash_6m

Generated 2026-09-19T00:17.
Train stressed company-months only. Holdout excluded from fit and sample.
Brief map: this Y is the **45 → 65** recovery direction (who is improving after a dip).
Question 5 (why) = top features. Question 6 (lead time) = lag1/lag3 mass.

- rows used: 4000 (of 5648 train labeled)
- positives in full train labeled: 402
- X columns: 278 families ACDEFGH (never B)
- trees: 400

## Top 15 by mean |SHAP|

| rank | feature | mean\|SHAP\| | family | brief bucket |
| --- | --- | --- | --- | --- |
| 1 | `a_transfer_lag1` | 0.35014 | a | lead_time / turning |
| 2 | `c_ss_month` | 0.309 | c | why (ops regularity) |
| 3 | `e_dso_proxy` | 0.29852 | e | why (counterparties / invoices) |
| 4 | `c_salary_month` | 0.23261 | c | why (ops regularity) |
| 5 | `a_transfer` | 0.22032 | a | other |
| 6 | `a_transfer_lag3` | 0.21116 | a | lead_time / turning |
| 7 | `h_n_siblings_active_lag3` | 0.211 | h | lead_time / turning |
| 8 | `c_gap_sd` | 0.20094 | c | why (ops regularity) |
| 9 | `f_ds_r` | 0.19648 | f | why (financing) |
| 10 | `c_n_days_with_tx` | 0.15576 | c | other |
| 11 | `a_n_tx` | 0.15113 | a | why (ops regularity) |
| 12 | `f_fc_r` | 0.14659 | f | why (financing) |
| 13 | `h_share_group_in` | 0.14221 | h | other |
| 14 | `h_group_size_lag3` | 0.13644 | h | lead_time / turning |
| 15 | `c_gap_sd_lag3` | 0.13151 | c | lead_time / turning |

## Family share of mean |SHAP|

| family | sum mean\|SHAP\| |
| --- | --- |
| a | 3.9714 |
| e | 2.101 |
| c | 1.5937 |
| h | 1.4964 |
| d | 0.85583 |
| f | 0.85357 |
| g | 0.32194 |
| group | 0.074491 |
| n | 0.074018 |

## Brief-question share of mean |SHAP|

| bucket | sum mean\|SHAP\| |
| --- | --- |
| lead_time / turning | 6.4201 |
| other | 2.2415 |
| why (ops regularity) | 0.94462 |
| why (counterparties / invoices) | 0.84356 |
| why (financing) | 0.45889 |
| healthy vs stressed cash shape | 0.43361 |

Plots: `shap_y3_summary_bar.png`, `shap_y3_beeswarm.png`. Table: `shap_y3_meanabs.csv`.

Do not treat holdout AUROC as confirmation. Quote train CV (0.710) plus this SHAP.
