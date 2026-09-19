# SHAP — y7_top1_lost

Generated 2026-09-19T00:24.
Train labeled company-months only. Holdout excluded from fit and sample.
Brief map: this Y is **dip vs fall** (question 4) — the current top AR customer
issues 0 in the next quarter. It is **not** bankruptcy.
Question 5 (why) = contemporaneous features. Question 6 (lead time) = lag1/lag3 mass.

- rows used: 4000 (of 7464 train labeled)
- positives in full train labeled: 2149
- train base rate: 0.2879
- X columns: 278 families ABCEFGH (never D)
- trees: 50
- store: parquet / Y: targets_parquet
- importance: mean |TreeSHAP| (shap 0.52.0 TreeExplainer)
- this-run group-fold CV AUROC: 0.663 (published bake-off **0.663**; holdout 0.680 is not the claim)
- Sign `+` = higher feature → higher P(top-1 lost).

## Top 15 by mean |SHAP|

| rank | feature | sign | mean\|SHAP\| | family | brief bucket |
| --- | --- | :---: | ---: | --- | --- |
| 1 | `e_dso_proxy` | + | 0.21585 | e | why (counterparties / invoices) |
| 2 | `e_ar_issued` | − | 0.21584 | e | why (counterparties / invoices) |
| 3 | `e_ar_issued_lag1` | − | 0.13794 | e | lead_time / turning |
| 4 | `f_fc_r_lag3` | + | 0.11293 | f | lead_time / turning |
| 5 | `e_credit_note_ratio_lag1` | + | 0.11053 | e | lead_time / turning |
| 6 | `e_delay_coll` | + | 0.081902 | e | why (counterparties / invoices) |
| 7 | `n_banking` | + | 0.07715 | meta | other |
| 8 | `e_credit_note_ratio` | + | 0.07603 | e | why (counterparties / invoices) |
| 9 | `e_dso_proxy_lag1` | + | 0.058761 | e | lead_time / turning |
| 10 | `group_size` | + | 0.055276 | meta | other |
| 11 | `e_ar_overdue` | + | 0.045449 | e | why (counterparties / invoices) |
| 12 | `e_ar_overdue_lag1` | + | 0.044274 | e | lead_time / turning |
| 13 | `e_delay_coll_lag1` | + | 0.036651 | e | lead_time / turning |
| 14 | `h_n_siblings_active` | + | 0.035376 | h | other |
| 15 | `e_delay_paid` | − | 0.032925 | e | why (counterparties / invoices) |

### What each of the top 10 is (questions 5–6)

1. `e_dso_proxy` (+) — AR open / this-period AR issued. Brief Q5: why (counterparties / invoices).
2. `e_ar_issued` (−) — AR issuance volume this period. Brief Q5: why (counterparties / invoices).
3. `e_ar_issued_lag1` (−) — AR issuance volume this period (t−1). Brief Q6: lead_time / turning.
4. `f_fc_r_lag3` (+) — trailing finance-cost / inflow (t−3). Brief Q6: lead_time / turning.
5. `e_credit_note_ratio_lag1` (+) — credit-notes / (invoices + credit-notes) (t−1). Brief Q6: lead_time / turning.
6. `e_delay_coll` (+) — amount-weighted AR collection delay (days). Brief Q5: why (counterparties / invoices).
7. `n_banking` (+) — count of banking products (snapshot). Brief Q5: other.
8. `e_credit_note_ratio` (+) — credit-notes / (invoices + credit-notes). Brief Q5: why (counterparties / invoices).
9. `e_dso_proxy_lag1` (+) — AR open / this-period AR issued (t−1). Brief Q6: lead_time / turning.
10. `group_size` (+) — companies in the group (static). Brief Q5: other.

### Notes on signs

- Sign is Spearman(feature, TreeSHAP) on the train sample. Univariate AUROC is a check.
- Sign disagreement (report SHAP): `e_dso_proxy_lag1` SHAP + vs univ − 0.506.

## Family share of mean |SHAP|

| family | sum mean\|SHAP\| |
| --- | ---: |
| e | 1.4289 |
| h | 0.27814 |
| a | 0.24909 |
| f | 0.16823 |
| b | 0.14543 |
| meta | 0.13243 |
| c | 0.11446 |
| g | 0.025046 |

## Brief-question share of mean |SHAP|

| bucket | sum mean\|SHAP\| |
| --- | ---: |
| lead_time / turning | 1.3158 |
| why (counterparties / invoices) | 0.75593 |
| other | 0.29709 |
| why (ops regularity) | 0.076298 |
| healthy vs stressed cash shape | 0.073307 |
| why (financing) | 0.023275 |

## Size proxy and D-concentration leak

Y7 vs `log1p(a_in3)` AUROC on train labeled = 0.465 (acceptance gate is < 0.60; the label is not a size clone).
X families **A+B+C+E+F+G+H**, never D (invoice HHI / top-1 are rebuilt inside Y7). n_x=278 including lags [1, 3].
Worst |ρ| vs `d_cust_top1` on train labeled (diagnostic, D is not X): `e_credit_note_ratio_lag1` ρ=-0.299.
No allowed X column reaches |ρ|≥0.80 vs `d_cust_top1` or `d_cust_hhi`. Family E (volume, DSO, overdue) is allowed and is not a rebuilt HHI.

## Judge paragraph — questions 5–6

`y7_top1_lost` is the brief's dip-vs-fall event (question 4): the current top AR customer issues nothing in t+1..t+3. It is not bankruptcy. Family D never enters X. Question 5 (why it changed) and question 6 (how many months earlier) are the SHAP object. Top 10: `e_dso_proxy` +, `e_ar_issued` −, `e_ar_issued_lag1` −, `f_fc_r_lag3` +, `e_credit_note_ratio_lag1` +, `e_delay_coll` +, `n_banking` +, `e_credit_note_ratio` +, `e_dso_proxy_lag1` +, `group_size` +.  Invoice-book volume/timing (family E, allowed) is the contemporaneous why: `e_dso_proxy`, `e_ar_issued`, `e_delay_coll`, `e_credit_note_ratio`.  Lead time (question 6) sits in `e_ar_issued_lag1`, `f_fc_r_lag3`, `e_credit_note_ratio_lag1`, `e_dso_proxy_lag1`.  Family B is allowed and present in X, but it is not a top-10 name. Across all X, lead-time / turning holds 52% of mean |SHAP|; explicit why-buckets (invoices + ops + financing) hold 34%; cash-shape residual is 0.0733. Family shares are led by e=1.43, h=0.278, a=0.249, f=0.168. No |ρ|≥0.85 size clone in the top 15. No top feature copies `d_cust_top1` / `d_cust_hhi` at |ρ|≥0.80; D stayed out of X. Quote train group-fold CV AUROC **0.663** (this run 0.663). Holdout 0.680 / 122 events is LOW_POWER — not the story.

Plots: `shap_y7_summary_bar.png`, `shap_y7_beeswarm.png`. Table: `shap_y7_meanabs.csv`.

Do not treat holdout AUROC 0.680 as confirmation. Quote train CV (0.663) plus this SHAP.

## Re-run

```bash
/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.explain_y7
```
