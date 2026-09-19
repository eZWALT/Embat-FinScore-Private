# Feature evaluation report (train only)

Holdout `analysis/splits/holdout_companies.csv` (72 companies) is **excluded** from every number below. No percentiles, bins, cluster centroids, or scalers were fit on holdout.

- Panel: **21,157** company-months, **1,214** train companies (monthly grid 2024-09 … 2026-08).
- Families loaded: **A,B,C,D,E,F,G,H** (102 numeric features).
- Families skipped: none.
- Size proxy: Spearman of each feature vs `log1p(|a_op_in|)`. Flag `|ρ| > 0.85`. The **model** size control is `log1p(a_in3)` (trailing 3m; median acf1 ≈ 0.66). Raw `a_op_in` has median Pearson acf1 ≈ 0 — monthly extremes, not a stable scale.
- Near-zero variance: modal-value share `≥ 0.95` (constant if `≥ 0.999`).
- Persistence: Pearson autocorr at lags 1 / 3 / 6, company-wise, then the median across companies (need ≥ 4 finite pairs and non-zero s.d.).
- Variance split: ANOVA `var_within` / `var_between`; ICC = between / (between + within). ICC `≥ 0.85` is flagged BETWEEN (mostly a company identity).
- Correlation clusters: average-linkage on `1 − |ρ|`, cut at `|ρ| = 0.8` (features with coverage `< 20.0%` excluded from the matrix).
- Company clustering: k-means `k = 4…8`, seed `20260918`, on standardised company medians. Silhouette picks k. Clusters are checked against log-inflow tertiles and `group_id`.

## Headline

### Top size-proxies

Ranked by `|Spearman ρ|` vs `log1p(|a_op_in|)`. Only rows with `|ρ| > 0.85` are *flagged* SIZE; the table shows the top 12 so weaker level-amount leaks are visible too.

| feature | ρ | |ρ| | flagged |
| --- | --- | --- | --- |
| a_op_in | 0.996 | 0.996 | SIZE |
| a_in3 | 0.875 | 0.875 | SIZE |
| a_in6 | 0.809 | 0.809 |  |
| h_share_group_in | 0.792 | 0.792 |  |
| a_op_out | 0.738 | 0.738 |  |
| a_in12 | 0.736 | 0.736 |  |
| c_n_tx | 0.706 | 0.706 |  |
| a_n_tx | 0.706 | 0.706 |  |
| a_out3 | 0.701 | 0.701 |  |
| a_out6 | 0.673 | 0.673 |  |
| c_n_days_with_tx | 0.647 | 0.647 |  |
| a_out12 | 0.630 | 0.630 |  |

Flagged SIZE (`|ρ| > 0.85`): `a_op_in`, `a_in3`.

### Near-zero variance / constants

| feature | modal% | n_unique | flags | rare_event |
| --- | --- | --- | --- | --- |
| g_created_unknown_share | 100.0% | 1 | CONSTANT |  |
| g_created_after_snapshot | 100.0% | 1 | CONSTANT |  |
| g_created_after_snapshot_share | 100.0% | 1 | CONSTANT |  |
| a_pending_share | 97.3% | 439 | NZV,BETWEEN,LOW_PERSIST |  |
| d_interco_share | — | 0 | CONSTANT,LOWCOV |  |

Rare-event flags (payroll miss, NSF-like `b_below_0`, product presence, `outstanding_gt_granted`) are **kept** even when the modal share is high: the minority class is the signal.

### Company-cluster silhouette

Best **k = 4**, silhouette **0.234** on **1214** companies using `a_io_ratio`, `a_growth_3`, `b_runway`, `b_bal_vol`, `c_zero_in_share_6`, `c_gap_sd`, `d_cust_hhi`, `d_tx_cp_share`, `e_ap_overdue_30`, `e_ar_overdue_30`, `f_ds_r`, `f_fc_r`.

| k | silhouette | chosen |
| --- | --- | --- |
| 4 | 0.234 | yes |
| 5 | 0.183 |  |
| 6 | 0.195 |  |
| 7 | 0.186 |  |
| 8 | 0.194 |  |

Cramér's V vs log-inflow tertile: **0.396** (NMI 0.171). NMI vs `group_id`: **0.112**. Among 138 groups with ≥ 3 train companies, 35 are mono-cluster (purity 25.4%).

Read: weak / overlapping structure — per-cluster models are not justified on this profile set. Cramér's V ≥ 0.35 plus a one-tertile pocket means part of the partition is a size cut — do not treat these as operating types.

### Recommended drop list

| feature | reason |
| --- | --- |
| a_debt_service | redundant with f_ds_r (|ρ| cluster cut 0.8) |
| a_in12 | redundant with a_in3 (|ρ| cluster cut 0.8) |
| a_n_tx | redundant with c_gap_sd (|ρ| cluster cut 0.8) |
| a_net | identity a_op_in − a_op_out (VIF explodes with the two levels) |
| a_net_margin | redundant with a_io_ratio (|ρ| cluster cut 0.8) |
| a_op_in | raw monthly inflow: SIZE vs its own log and median acf1≈0 (extremes); use log1p(a_in3) |
| a_out12 | redundant with a_op_out (|ρ| cluster cut 0.8) |
| a_out3 | redundant with a_op_out (|ρ| cluster cut 0.8) |
| a_out6 | redundant with a_op_out (|ρ| cluster cut 0.8) |
| a_pending_share | near-zero variance (modal share 97.3%) |
| b_below_half_runway | redundant with b_runway (|ρ| cluster cut 0.8) |
| b_mean_liq_3 | redundant with b_liq (|ρ| cluster cut 0.8) |
| b_min_liq_3 | redundant with b_liq (|ρ| cluster cut 0.8) |
| b_neg_liq_3 | redundant with b_below_0 (|ρ| cluster cut 0.8) |
| c_n_days_with_tx | redundant with c_gap_sd (|ρ| cluster cut 0.8) |
| c_n_tx | redundant with c_gap_sd (|ρ| cluster cut 0.8) |
| d_cust_lost | redundant with d_n_cust (|ρ| cluster cut 0.8) |
| d_cust_top1 | redundant with d_cust_hhi (|ρ| cluster cut 0.8) |
| d_interco_share | constant or all-null (no variance) |
| d_supp_top1 | redundant with d_supp_hhi (|ρ| cluster cut 0.8) |
| e_ap_overdue | redundant with e_ap_overdue_30 (|ρ| cluster cut 0.8) |
| f_debt_service | redundant with f_ds_r (|ρ| cluster cut 0.8) |
| f_fin_cost | redundant with a_fin_cost (|ρ| cluster cut 0.8) |
| f_n_facilities | redundant with f_n_types (|ρ| cluster cut 0.8) |
| g_created_after_snapshot | constant or all-null (no variance) |
| g_created_after_snapshot_share | constant or all-null (no variance) |
| g_created_unknown_share | constant or all-null (no variance) |
| g_n_banks | redundant with g_n_accounts (|ρ| cluster cut 0.8) |
| h_group_size | redundant with h_n_siblings_active (|ρ| cluster cut 0.8) |
| h_sib_in | redundant with h_sib_out (|ρ| cluster cut 0.8) |

### Park (sparse, near-size, or raw level — not in the core GBM set)

| feature | reason |
| --- | --- |
| a_fin_cost | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |
| a_in6 | near-size |ρ|=0.809 vs log inflow — do not add besides a_in3 |
| a_invest | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |
| a_op_out | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |
| a_transfer | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |
| b_liq | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |
| d_cust_new | company-median VIF=19.4 with d_n_cust |
| e_ap_issued | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |
| e_ap_open | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |
| e_ar_issued | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |
| e_ar_open | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |
| f_months_to_next_pay | coverage 1.7% — last-month / sparse only |
| f_sched_vs_obs | coverage 1.7% — last-month / sparse only |
| f_util_snapshot | coverage 1.6% — last-month / sparse only |
| f_w_rate | coverage 1.7% — last-month / sparse only |
| g_n_types | company-median VIF=22.9 with product-mix flags |
| h_share_group_in | near-size |ρ|=0.792 vs log inflow — do not add besides a_in3 |
| h_sib_net | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |
| h_sib_out | raw level amount — singleton cluster on the panel, but company-median VIF collinear with size; log if needed |

### Recommended keep

| feature | reason |
| --- | --- |
| a_growth_12 | cluster representative (share / ratio / flag) |
| a_growth_3 | cluster representative (share / ratio / flag) |
| a_in3 | canonical size control — trailing 3m inflow (use log1p); more persistent than a_op_in |
| a_io_ratio | cluster representative (share / ratio / flag) |
| a_uncat_share | cluster representative (share / ratio / flag) |
| b_bal_vol | cluster representative (share / ratio / flag) |
| b_below_0 | rare-event flag (modal 92.2%) |
| b_d_runway | cluster representative (share / ratio / flag) |
| b_neg_episodes | rare-event flag (modal 92.7%) |
| b_runway | cluster representative (share / ratio / flag) |
| c_gap_sd | cluster representative (share / ratio / flag) |
| c_last_tx_before_2026_06 | rare-event flag (modal 99.0%) |
| c_missed_salary | rare-event flag (modal 97.1%) |
| c_missed_tax | rare-event flag (modal 91.7%) |
| c_recency_days | cluster representative (share / ratio / flag) |
| c_salary_month | cluster representative (share / ratio / flag) |
| c_ss_month | cluster representative (share / ratio / flag) |
| c_tax_month | cluster representative (share / ratio / flag) |
| c_zero_in_month | rare-event flag (modal 88.2%) |
| c_zero_in_share_6 | cluster representative (share / ratio / flag) |
| d_cust_hhi | cluster representative (share / ratio / flag) |
| d_n_cust | cluster representative (share / ratio / flag) |
| d_n_supp | cluster representative (share / ratio / flag) |
| d_supp_hhi | cluster representative (share / ratio / flag) |
| d_tx_cp_share | cluster representative (share / ratio / flag) |
| e_ap_overdue_30 | cluster representative (share / ratio / flag) |
| e_ar_overdue | cluster representative (share / ratio / flag) |
| e_ar_overdue_30 | cluster representative (share / ratio / flag) |
| e_credit_note_ratio | cluster representative (share / ratio / flag) |
| e_delay_coll | cluster representative (share / ratio / flag) |
| e_delay_paid | cluster representative (share / ratio / flag) |
| e_dpo_proxy | cluster representative (share / ratio / flag) |
| e_dso_proxy | cluster representative (share / ratio / flag) |
| e_fx_share | cluster representative (share / ratio / flag) |
| e_pending_amt_share | cluster representative (share / ratio / flag) |
| f_ds_r | cluster representative (share / ratio / flag) |
| f_fc_r | cluster representative (share / ratio / flag) |
| f_has_confirming | rare-event flag (modal 96.7%) |
| f_has_factoring | rare-event flag (modal 99.3%) |
| f_has_loc | rare-event flag (modal 88.3%) |
| f_n_types | cluster representative (share / ratio / flag) |
| f_new_facility | rare-event flag (modal 97.3%) |
| f_outstanding_gt_granted | rare-event flag (modal 97.4%) |
| g_custom_share | cluster representative (share / ratio / flag) |
| g_has_card | rare-event flag (modal 86.8%) |
| g_has_checking | rare-event flag (modal 87.5%) |
| g_has_investment | rare-event flag (modal 95.1%) |
| g_has_saving | rare-event flag (modal 99.7%) |
| g_has_tpv | rare-event flag (modal 99.6%) |
| g_n_accounts | cluster representative (share / ratio / flag) |
| g_new_this_month | cluster representative (share / ratio / flag) |
| h_n_siblings_active | cluster representative (share / ratio / flag) |
| h_sib_neg_share | cluster representative (share / ratio / flag) |

Core GBM starter set (size + shares/ratios/flags from the keep list): `a_growth_12`, `a_growth_3`, `a_in3`, `a_io_ratio`, `a_uncat_share`, `b_bal_vol`, `b_below_0`, `b_d_runway`, `b_neg_episodes`, `b_runway`, `c_gap_sd`, `c_last_tx_before_2026_06`, `c_missed_salary`, `c_missed_tax`, `c_recency_days`, `c_zero_in_month`, `c_zero_in_share_6`, `d_cust_hhi`, `d_supp_hhi`, `d_tx_cp_share`, `e_ap_overdue_30`, `e_ar_overdue`, `e_ar_overdue_30`, `e_credit_note_ratio`, `e_delay_coll`, `e_delay_paid`, `e_dpo_proxy`, `e_dso_proxy`, `e_fx_share`, `e_pending_amt_share`, `f_ds_r`, `f_fc_r`, `f_has_confirming`, `f_has_factoring`, `f_has_loc`, `f_new_facility`, `f_outstanding_gt_granted`, `g_custom_share`, `g_has_card`, `g_has_checking`, `g_has_investment`, `g_has_saving`, `g_has_tpv`, `h_sib_neg_share`.

## 1. Coverage by family

| family | n_cols | median cov_cm | min cov_cm | median ICC | n SIZE | n NZV/CONST |
| --- | --- | --- | --- | --- | --- | --- |
| a | 20 | 91.8% | 26.5% | 0.95 | 2 | 1 |
| b | 10 | 87.7% | 70.7% | 0.96 | 0 | 0 |
| c | 12 | 100.0% | 95.1% | 0.95 | 0 | 0 |
| d | 10 | 51.7% | 0.0% | 0.98 | 0 | 1 |
| e | 15 | 53.0% | 31.9% | 0.94 | 0 | 0 |
| f | 15 | 100.0% | 1.6% | 0.97 | 0 | 0 |
| g | 13 | 100.0% | 87.6% | 0.98 | 0 | 3 |
| h | 7 | 100.0% | 94.9% | 0.96 | 0 | 0 |

`d_interco_share` is all-null on this extract (no usable intercompany IDs). Family F snapshot fields (`f_util_snapshot`, `f_w_rate`, `f_months_to_next_pay`, `f_sched_vs_obs`) are populated only near the 2026-09-01 book — they are not a panel series.

## 2. Per-feature battery

Flags: `SIZE` `|ρ|>0.85` vs log inflow; `NZV` modal share ≥ 0.95; `CONSTANT` modal share ≥ 0.999 or a single value; `BETWEEN` ICC ≥ 0.85; `LOWCOV` company-month coverage < 5%; `LOW_PERSIST` median company acf1 `|r|<0.25` on a level (not a change feature); `RARE` = rare-event keep.

### Family A — cash flow

| feature | cov_cm | cov_co | modal% | size_ρ | acf1 | acf3 | acf6 | w/b | ICC | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_debt_service | 100.0% | 100.0% | 76.4% | 0.333 | 0.03 | -4.02e-17 | -0.03 | 0.33 | 0.75 | LOW_PERSIST |
| a_fin_cost | 100.0% | 100.0% | 43.5% | 0.446 | -0.06 | -0.01 | -0.05 | 0.33 | 0.75 | LOW_PERSIST |
| a_growth_12 | 26.5% | 59.9% | 20.4% | 0.451 | 0.53 | -0.34 | 0.05 | 0.11 | 0.90 | BETWEEN |
| a_growth_3 | 64.0% | 97.9% | 18.3% | 0.319 | 0.52 | -0.43 | -0.07 | 0.42 | 0.70 | — |
| a_in12 | 41.0% | 71.2% | 2.7% | 0.736 | 0.83 | 0.51 | 0.35 | 0.02 | 0.98 | BETWEEN |
| a_in3 | 88.5% | 100.0% | 10.4% | 0.875 | 0.66 | -0.09 | -0.12 | 0.04 | 0.96 | SIZE,BETWEEN |
| a_in6 | 71.3% | 99.8% | 6.4% | 0.809 | 0.80 | 0.33 | -0.36 | 0.04 | 0.96 | BETWEEN |
| a_invest | 100.0% | 100.0% | 89.6% | -0.015 | -0.06 | -0.05 | -0.07 | 1.60 | 0.38 | LOW_PERSIST |
| a_io_ratio | 88.5% | 100.0% | 14.9% | 0.319 | 0.55 | -0.14 | -0.09 | 0.43 | 0.70 | — |
| a_n_tx | 100.0% | 100.0% | 4.2% | 0.706 | 0.22 | 0.13 | 0.07 | 0.03 | 0.97 | BETWEEN,LOW_PERSIST |
| a_net | 100.0% | 100.0% | 8.8% | 0.232 | -0.05 | -0.00 | -0.02 | 0.19 | 0.84 | LOW_PERSIST |
| a_net_margin | 88.5% | 100.0% | 27.3% | 0.276 | 0.55 | -0.14 | -0.10 | 0.07 | 0.93 | BETWEEN |
| a_op_in | 100.0% | 100.0% | 19.9% | 0.996 | 0.00 | -0.01 | -0.05 | 0.06 | 0.95 | SIZE,BETWEEN,LOW_PERSIST |
| a_op_out | 100.0% | 100.0% | 12.7% | 0.738 | 0.01 | 0.04 | -0.00 | 0.05 | 0.96 | BETWEEN,LOW_PERSIST |
| a_out12 | 41.0% | 71.2% | 1.1% | 0.630 | 0.84 | 0.53 | 0.43 | 0.03 | 0.97 | BETWEEN |
| a_out3 | 88.5% | 100.0% | 6.3% | 0.701 | 0.68 | -0.06 | -0.12 | 0.04 | 0.96 | BETWEEN |
| a_out6 | 71.3% | 99.8% | 3.5% | 0.673 | 0.81 | 0.35 | -0.33 | 0.05 | 0.96 | BETWEEN |
| a_pending_share | 95.1% | 99.7% | 97.3% | 0.121 | -0.05 | -0.07 | -0.06 | 0.10 | 0.91 | NZV,BETWEEN,LOW_PERSIST |
| a_transfer | 100.0% | 100.0% | 62.4% | 0.021 | -0.03 | -0.05 | -0.06 | 0.05 | 0.96 | BETWEEN,LOW_PERSIST |
| a_uncat_share | 95.8% | 100.0% | 21.1% | -0.023 | 0.27 | 0.12 | 0.04 | 0.01 | 0.99 | BETWEEN |

### Family B — liquidity

| feature | cov_cm | cov_co | modal% | size_ρ | acf1 | acf3 | acf6 | w/b | ICC | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b_bal_vol | 70.7% | 98.2% | 3.2% | -0.270 | 0.67 | 0.10 | -0.22 | 0.58 | 0.63 | — |
| b_below_0 | 99.0% | 98.4% | 92.2% | 0.038 | 0.41 | -0.05 | -0.02 | 0.04 | 0.96 | BETWEEN |
| b_below_half_runway | 87.7% | 98.4% | 61.9% | 0.220 | 0.34 | -0.08 | -0.10 | 0.04 | 0.96 | BETWEEN |
| b_d_runway | 70.7% | 98.2% | 13.4% | -0.036 | 0.44 | -0.44 | -0.04 | 0.91 | 0.52 | — |
| b_liq | 99.0% | 98.4% | 1.1% | 0.344 | 0.48 | 0.16 | 0.01 | 0.04 | 0.96 | BETWEEN |
| b_mean_liq_3 | 87.7% | 98.4% | 0.8% | 0.345 | 0.84 | 0.26 | -0.02 | 0.03 | 0.97 | BETWEEN |
| b_min_liq_3 | 87.7% | 98.4% | 1.1% | 0.266 | 0.73 | 0.22 | -0.03 | 0.06 | 0.94 | BETWEEN |
| b_neg_episodes | 70.7% | 98.2% | 92.7% | 0.068 | 0.75 | 0.35 | -0.46 | 0.07 | 0.94 | BETWEEN |
| b_neg_liq_3 | 87.7% | 98.4% | 89.3% | 0.065 | 0.81 | 0.04 | -0.05 | 0.03 | 0.97 | BETWEEN |
| b_runway | 87.7% | 98.4% | 16.9% | -0.305 | 0.51 | -0.02 | -0.05 | 0.03 | 0.97 | BETWEEN |

### Family C — operations

| feature | cov_cm | cov_co | modal% | size_ρ | acf1 | acf3 | acf6 | w/b | ICC | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c_gap_sd | 95.1% | 100.0% | 4.5% | -0.543 | 0.61 | -0.03 | -0.07 | 0.04 | 0.96 | BETWEEN |
| c_last_tx_before_2026_06 | 100.0% | 100.0% | 99.0% | -0.137 | 0.79 | — | — | 0.37 | 0.73 | RARE |
| c_missed_salary | 100.0% | 100.0% | 97.1% | 0.024 | 0.34 | -0.14 | -0.17 | 0.36 | 0.74 | RARE |
| c_missed_tax | 100.0% | 100.0% | 91.7% | -0.017 | 0.05 | 0.07 | -0.11 | 0.39 | 0.72 | — |
| c_n_days_with_tx | 100.0% | 100.0% | 5.6% | 0.647 | 0.13 | 0.11 | 0.07 | 0.02 | 0.99 | BETWEEN,LOW_PERSIST |
| c_n_tx | 100.0% | 100.0% | 4.2% | 0.706 | 0.22 | 0.13 | 0.07 | 0.03 | 0.97 | BETWEEN,LOW_PERSIST |
| c_recency_days | 100.0% | 100.0% | 58.5% | -0.447 | -0.19 | 0.25 | 0.32 | 0.08 | 0.93 | BETWEEN,LOW_PERSIST |
| c_salary_month | 100.0% | 100.0% | 59.9% | 0.350 | 0.09 | -0.05 | -0.08 | 0.02 | 0.98 | BETWEEN,LOW_PERSIST |
| c_ss_month | 100.0% | 100.0% | 54.3% | 0.362 | 0.60 | 0.14 | -0.06 | 0.01 | 0.99 | BETWEEN |
| c_tax_month | 100.0% | 100.0% | 51.8% | 0.378 | -0.13 | 0.28 | 0.25 | 0.08 | 0.93 | BETWEEN,LOW_PERSIST |
| c_zero_in_month | 100.0% | 100.0% | 88.2% | -0.508 | -0.03 | -1.07e-17 | -0.06 | 0.07 | 0.93 | BETWEEN,LOW_PERSIST |
| c_zero_in_share_6 | 100.0% | 100.0% | 77.4% | -0.519 | 0.87 | 0.60 | -0.19 | 0.03 | 0.97 | BETWEEN |

### Family D — counterparties

| feature | cov_cm | cov_co | modal% | size_ρ | acf1 | acf3 | acf6 | w/b | ICC | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d_cust_hhi | 42.2% | 55.5% | 15.6% | -0.203 | 0.77 | 0.27 | -0.30 | 0.02 | 0.98 | BETWEEN |
| d_cust_lost | 53.6% | 61.2% | 43.8% | 0.323 | 0.58 | 0.02 | -0.01 | 0.03 | 0.98 | BETWEEN |
| d_cust_new | 53.6% | 61.2% | 49.1% | 0.345 | 0.29 | 0.14 | 0.19 | 0.04 | 0.97 | BETWEEN |
| d_cust_top1 | 42.2% | 55.5% | 15.6% | -0.192 | 0.74 | 0.21 | -0.27 | 0.02 | 0.98 | BETWEEN |
| d_interco_share | 0.0% | 0.0% | — | — | — | — | — | — | — | CONSTANT,LOWCOV |
| d_n_cust | 53.4% | 61.3% | 20.9% | 0.396 | 0.87 | 0.54 | -0.06 | 0.02 | 0.98 | BETWEEN |
| d_n_supp | 53.4% | 61.3% | 6.2% | 0.530 | 0.91 | 0.64 | -0.06 | 0.04 | 0.96 | BETWEEN |
| d_supp_hhi | 50.1% | 61.0% | 4.9% | -0.351 | 0.75 | 0.28 | -0.26 | 0.04 | 0.96 | BETWEEN |
| d_supp_top1 | 50.1% | 61.0% | 4.9% | -0.318 | 0.71 | 0.23 | -0.29 | 0.04 | 0.96 | BETWEEN |
| d_tx_cp_share | 99.2% | 100.0% | 57.8% | 0.049 | 0.93 | 0.74 | 0.26 | 0.02 | 0.98 | BETWEEN |

### Family E — invoices

| feature | cov_cm | cov_co | modal% | size_ρ | acf1 | acf3 | acf6 | w/b | ICC | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e_ap_issued | 64.1% | 61.4% | 20.1% | 0.478 | 0.17 | 0.06 | -0.05 | 0.86 | 0.54 | LOW_PERSIST |
| e_ap_open | 64.1% | 61.3% | 10.2% | 0.411 | 0.74 | 0.45 | 0.21 | 0.01 | 0.99 | BETWEEN |
| e_ap_overdue | 57.5% | 61.0% | 18.7% | -0.202 | 0.51 | 0.16 | 0.03 | 0.05 | 0.95 | BETWEEN |
| e_ap_overdue_30 | 57.5% | 61.0% | 18.9% | -0.150 | 0.62 | 0.25 | 0.09 | 0.07 | 0.94 | BETWEEN |
| e_ar_issued | 64.1% | 61.4% | 36.7% | 0.454 | 0.14 | 0.09 | -0.00 | 0.06 | 0.94 | BETWEEN,LOW_PERSIST |
| e_ar_open | 64.1% | 61.3% | 27.5% | 0.387 | 0.77 | 0.50 | 0.30 | 0.01 | 0.99 | BETWEEN |
| e_ar_overdue | 46.4% | 54.9% | 33.9% | -0.214 | 0.34 | 0.13 | -0.04 | 0.06 | 0.95 | BETWEEN |
| e_ar_overdue_30 | 46.4% | 54.9% | 18.4% | -0.140 | 0.61 | 0.27 | 0.09 | 0.08 | 0.92 | BETWEEN |
| e_credit_note_ratio | 53.0% | 61.3% | 57.7% | 0.218 | -0.09 | -0.09 | -0.07 | 0.21 | 0.83 | LOW_PERSIST |
| e_delay_coll | 31.9% | 48.9% | 23.2% | -0.086 | 0.63 | -0.15 | -0.19 | 0.09 | 0.92 | BETWEEN |
| e_delay_paid | 40.7% | 56.4% | 22.5% | -0.041 | 0.60 | -0.19 | -0.21 | 0.07 | 0.93 | BETWEEN |
| e_dpo_proxy | 51.2% | 61.0% | 3.4% | -0.015 | 0.24 | 0.01 | 0.01 | 0.24 | 0.81 | LOW_PERSIST |
| e_dso_proxy | 40.6% | 55.4% | 6.5% | 0.034 | 0.25 | 0.07 | 0.01 | 0.59 | 0.63 | — |
| e_fx_share | 52.8% | 61.2% | 79.6% | 0.052 | -0.02 | -0.05 | -0.10 | 0.02 | 0.98 | BETWEEN,LOW_PERSIST |
| e_pending_amt_share | 60.3% | 61.3% | 12.0% | 0.044 | 0.78 | 0.48 | 0.31 | 0.03 | 0.97 | BETWEEN |

### Family F — debt

| feature | cov_cm | cov_co | modal% | size_ρ | acf1 | acf3 | acf6 | w/b | ICC | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f_debt_service | 100.0% | 100.0% | 76.4% | 0.333 | 0.03 | -4.02e-17 | -0.03 | 0.33 | 0.75 | LOW_PERSIST |
| f_ds_r | 88.5% | 100.0% | 71.2% | 0.274 | 0.62 | -0.09 | -0.11 | 0.05 | 0.95 | BETWEEN |
| f_fc_r | 88.5% | 100.0% | 28.8% | 0.073 | 0.58 | -0.13 | -0.17 | 0.07 | 0.94 | BETWEEN |
| f_fin_cost | 100.0% | 100.0% | 43.5% | 0.446 | -0.06 | -0.01 | -0.05 | 0.33 | 0.75 | LOW_PERSIST |
| f_has_confirming | 100.0% | 100.0% | 96.7% | 0.146 | 0.86 | 0.69 | 0.39 | 0.02 | 0.98 | RARE,BETWEEN |
| f_has_factoring | 100.0% | 100.0% | 99.3% | 0.076 | 0.78 | 0.74 | 0.48 | 0.03 | 0.97 | RARE,BETWEEN |
| f_has_loc | 100.0% | 100.0% | 88.3% | 0.240 | 0.69 | 0.70 | 0.40 | 0.01 | 0.99 | BETWEEN |
| f_months_to_next_pay | 1.7% | 3.1% | 2.4% | 0.044 | 1.00 | 1.00 | 1.00 | 0.09 | 0.92 | BETWEEN,LOWCOV |
| f_n_facilities | 100.0% | 100.0% | 77.1% | 0.292 | 0.85 | 0.73 | 0.53 | 0.03 | 0.98 | BETWEEN |
| f_n_types | 100.0% | 100.0% | 77.1% | 0.290 | 0.80 | 0.70 | 0.45 | 0.02 | 0.98 | BETWEEN |
| f_new_facility | 100.0% | 100.0% | 97.3% | 0.129 | -0.06 | -0.07 | -0.08 | 0.61 | 0.62 | RARE |
| f_outstanding_gt_granted | 5.7% | 100.0% | 97.4% | 0.092 | — | — | — | 0.00 | 1.00 | RARE,BETWEEN |
| f_sched_vs_obs | 1.7% | 3.1% | 31.2% | -0.335 | -0.08 | 0.11 | -0.02 | 0.08 | 0.92 | BETWEEN,LOWCOV,LOW_PERSIST |
| f_util_snapshot | 1.6% | 27.5% | 10.8% | -0.034 | — | — | — | 0.00 | 1.00 | BETWEEN,LOWCOV |
| f_w_rate | 1.7% | 3.1% | 19.6% | -0.252 | 0.30 | 0.27 | -0.26 | 9.21e-05 | 1.00 | BETWEEN,LOWCOV |

### Family G — products

| feature | cov_cm | cov_co | modal% | size_ρ | acf1 | acf3 | acf6 | w/b | ICC | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g_created_after_snapshot | 100.0% | 100.0% | 100.0% | — | — | — | — | — | — | CONSTANT |
| g_created_after_snapshot_share | 87.6% | 99.6% | 100.0% | — | — | — | — | — | — | CONSTANT |
| g_created_unknown_share | 87.6% | 99.6% | 100.0% | — | — | — | — | — | — | CONSTANT |
| g_custom_share | 87.6% | 99.6% | 90.2% | 0.017 | 0.84 | 0.46 | 0.40 | 0.00 | 1.00 | BETWEEN |
| g_has_card | 100.0% | 100.0% | 86.8% | 0.155 | 0.77 | 0.68 | 0.39 | 0.01 | 0.99 | BETWEEN |
| g_has_checking | 100.0% | 100.0% | 87.5% | 0.037 | 0.68 | 0.66 | 0.39 | 0.13 | 0.89 | BETWEEN |
| g_has_investment | 100.0% | 100.0% | 95.1% | 0.170 | 0.80 | 0.68 | 0.44 | 0.02 | 0.98 | RARE,BETWEEN |
| g_has_saving | 100.0% | 100.0% | 99.7% | -0.011 | 0.64 | 0.64 | — | 0.02 | 0.98 | RARE,BETWEEN |
| g_has_tpv | 100.0% | 100.0% | 99.6% | 0.046 | 0.90 | 0.68 | 0.35 | 0.04 | 0.96 | RARE,BETWEEN |
| g_n_accounts | 100.0% | 100.0% | 24.1% | 0.353 | 0.80 | 0.70 | 0.50 | 0.02 | 0.98 | BETWEEN |
| g_n_banks | 100.0% | 100.0% | 37.6% | 0.321 | 0.75 | 0.70 | 0.50 | 0.02 | 0.98 | BETWEEN |
| g_n_types | 100.0% | 100.0% | 69.9% | 0.160 | 0.69 | 0.68 | 0.40 | 0.05 | 0.95 | BETWEEN |
| g_new_this_month | 100.0% | 100.0% | 92.1% | 0.077 | -0.08 | -0.08 | -0.08 | 0.63 | 0.62 | — |

### Family H — group context

| feature | cov_cm | cov_co | modal% | size_ρ | acf1 | acf3 | acf6 | w/b | ICC | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h_group_size | 100.0% | 100.0% | 8.1% | -0.016 | — | — | — | 0.00 | 1.00 | BETWEEN |
| h_n_siblings_active | 100.0% | 100.0% | 7.4% | -0.004 | 0.77 | 0.48 | 0.27 | 0.00 | 1.00 | BETWEEN |
| h_share_group_in | 98.0% | 99.9% | 18.2% | 0.792 | 0.01 | -0.03 | -0.04 | 0.02 | 0.98 | BETWEEN,LOW_PERSIST |
| h_sib_in | 100.0% | 100.0% | 7.3% | 0.109 | 0.03 | 0.03 | 0.09 | 0.05 | 0.95 | BETWEEN,LOW_PERSIST |
| h_sib_neg_share | 94.9% | 94.7% | 10.5% | 0.055 | 0.05 | 0.08 | 0.12 | 0.11 | 0.90 | BETWEEN,LOW_PERSIST |
| h_sib_net | 100.0% | 100.0% | 6.0% | -0.018 | -0.03 | 0.03 | 0.04 | 0.12 | 0.89 | BETWEEN,LOW_PERSIST |
| h_sib_out | 100.0% | 100.0% | 6.4% | 0.107 | 0.10 | 0.19 | 0.13 | 0.04 | 0.96 | BETWEEN,LOW_PERSIST |

## 3. Correlation clusters (`|ρ|` cut 0.80)

93 features entered the Spearman matrix; 68 clusters; 17 clusters have more than one member. One representative is kept per cluster (highest score: coverage, not-size, not-NZV, share/ratio names; `a_op_in` wins a size cluster).

| cluster | n | representative | members | max_abs_rho |
| --- | --- | --- | --- | --- |
| 1 | 1 | c_missed_salary | c_missed_salary | 0.00 |
| 2 | 1 | c_missed_tax | c_missed_tax | 0.00 |
| 3 | 2 | h_n_siblings_active | h_n_siblings_active, h_group_size | 0.98 |
| 4 | 2 | h_sib_out | h_sib_out, h_sib_in | 0.92 |
| 5 | 1 | h_sib_net | h_sib_net | 0.00 |
| 6 | 1 | h_sib_neg_share | h_sib_neg_share | 0.00 |
| 7 | 1 | f_new_facility | f_new_facility | 0.00 |
| 8 | 1 | g_new_this_month | g_new_this_month | 0.00 |
| 9 | 1 | g_has_tpv | g_has_tpv | 0.00 |
| 10 | 1 | e_delay_coll | e_delay_coll | 0.00 |
| 11 | 1 | e_delay_paid | e_delay_paid | 0.00 |
| 12 | 1 | e_dso_proxy | e_dso_proxy | 0.00 |
| 13 | 1 | e_pending_amt_share | e_pending_amt_share | 0.00 |
| 14 | 1 | e_ar_overdue | e_ar_overdue | 0.00 |
| 15 | 1 | e_ar_overdue_30 | e_ar_overdue_30 | 0.00 |
| 16 | 2 | e_ap_overdue_30 | e_ap_overdue_30, e_ap_overdue | 0.85 |
| 17 | 1 | e_dpo_proxy | e_dpo_proxy | 0.00 |
| 18 | 1 | a_uncat_share | a_uncat_share | 0.00 |
| 19 | 1 | e_fx_share | e_fx_share | 0.00 |
| 20 | 1 | g_custom_share | g_custom_share | 0.00 |
| 21 | 1 | d_tx_cp_share | d_tx_cp_share | 0.00 |
| 22 | 2 | a_io_ratio | a_io_ratio, a_net_margin | 0.99 |
| 23 | 1 | a_net | a_net | 0.00 |
| 24 | 1 | a_growth_3 | a_growth_3 | 0.00 |
| 25 | 1 | a_growth_12 | a_growth_12 | 0.00 |
| 26 | 1 | a_transfer | a_transfer | 0.00 |
| 27 | 1 | b_d_runway | b_d_runway | 0.00 |
| 28 | 2 | b_runway | b_runway, b_below_half_runway | 0.84 |
| 29 | 1 | b_bal_vol | b_bal_vol | 0.00 |
| 30 | 3 | b_liq | b_liq, b_min_liq_3, b_mean_liq_3 | 0.95 |
| 31 | 2 | b_below_0 | b_below_0, b_neg_liq_3 | 0.84 |
| 32 | 1 | b_neg_episodes | b_neg_episodes | 0.00 |
| 33 | 2 | d_cust_hhi | d_cust_hhi, d_cust_top1 | 0.99 |
| 34 | 2 | d_n_cust | d_n_cust, d_cust_lost | 0.86 |
| 35 | 1 | d_cust_new | d_cust_new | 0.00 |
| 36 | 1 | e_credit_note_ratio | e_credit_note_ratio | 0.00 |
| 37 | 2 | a_fin_cost | a_fin_cost, f_fin_cost | 1.00 |
| 38 | 1 | f_fc_r | f_fc_r | 0.00 |
| 39 | 2 | d_supp_hhi | d_supp_hhi, d_supp_top1 | 0.99 |
| 40 | 1 | e_ar_open | e_ar_open | 0.00 |
| 41 | 1 | e_ar_issued | e_ar_issued | 0.00 |
| 42 | 1 | d_n_supp | d_n_supp | 0.00 |
| 43 | 1 | e_ap_issued | e_ap_issued | 0.00 |
| 44 | 1 | e_ap_open | e_ap_open | 0.00 |
| 45 | 1 | c_zero_in_month | c_zero_in_month | 0.00 |
| 46 | 1 | c_zero_in_share_6 | c_zero_in_share_6 | 0.00 |
| 47 | 4 | c_gap_sd | c_gap_sd, c_n_days_with_tx, a_n_tx, c_n_tx | 1.00 |
| 48 | 4 | a_op_out | a_op_out, a_out3, a_out6, a_out12 | 0.95 |
| 49 | 4 | a_in3 | a_in3, a_in6, a_in12, a_op_in | 0.93 |
| 50 | 1 | h_share_group_in | h_share_group_in | 0.00 |
| 51 | 1 | c_recency_days | c_recency_days | 0.00 |
| 52 | 1 | c_salary_month | c_salary_month | 0.00 |
| 53 | 1 | c_ss_month | c_ss_month | 0.00 |
| 54 | 1 | c_tax_month | c_tax_month | 0.00 |
| 55 | 3 | f_ds_r | f_ds_r, a_debt_service, f_debt_service | 1.00 |
| 56 | 2 | f_n_types | f_n_types, f_n_facilities | 0.99 |
| 57 | 1 | f_has_loc | f_has_loc | 0.00 |
| 58 | 2 | g_n_accounts | g_n_accounts, g_n_banks | 0.88 |
| 59 | 1 | g_n_types | g_n_types | 0.00 |
| 60 | 1 | g_has_checking | g_has_checking | 0.00 |
| 61 | 1 | g_has_card | g_has_card | 0.00 |
| 62 | 1 | g_has_investment | g_has_investment | 0.00 |
| 63 | 1 | f_has_factoring | f_has_factoring | 0.00 |
| 64 | 1 | f_has_confirming | f_has_confirming | 0.00 |
| 65 | 1 | a_pending_share | a_pending_share | 0.00 |
| 66 | 1 | c_last_tx_before_2026_06 | c_last_tx_before_2026_06 | 0.00 |
| 67 | 1 | g_has_saving | g_has_saving | 0.00 |
| 68 | 1 | a_invest | a_invest | 0.00 |

### VIF on company-median representatives

VIF is computed on **train company medians** of *non-amount* cluster representatives with ≥ 50% company coverage, after standardising. Raw euro levels are left out on purpose: their median-scale VIF is 10³–10⁵ because size is one factor. VIF `> 10` among the remaining reps still wants a drop.

| feature | VIF |
| --- | --- |
| g_n_types | 22.86 |
| d_cust_new | 19.43 |
| d_n_cust | 19.42 |
| g_has_card | 11.93 |
| g_has_investment | 5.69 |
| f_n_types | 4.55 |
| c_zero_in_share_6 | 4.22 |
| e_ar_overdue_30 | 4.14 |
| e_ar_overdue | 4.04 |
| c_gap_sd | 3.67 |
| c_zero_in_month | 3.36 |
| g_has_checking | 3.11 |
| a_uncat_share | 2.72 |
| f_has_loc | 2.70 |
| c_ss_month | 2.69 |
| g_n_accounts | 2.63 |
| g_has_tpv | 2.48 |
| c_salary_month | 2.35 |
| b_runway | 2.27 |
| f_has_confirming | 2.22 |
| a_growth_3 | 1.89 |
| e_ap_overdue_30 | 1.89 |
| a_io_ratio | 1.88 |
| d_n_supp | 1.85 |
| h_share_group_in | 1.82 |
| d_supp_hhi | 1.77 |
| h_n_siblings_active | 1.75 |
| c_recency_days | 1.75 |
| d_cust_hhi | 1.72 |
| c_tax_month | 1.70 |
| a_growth_12 | 1.70 |
| g_new_this_month | 1.68 |
| e_credit_note_ratio | 1.66 |
| e_pending_amt_share | 1.62 |
| g_custom_share | 1.61 |
| e_delay_paid | 1.61 |
| d_tx_cp_share | 1.52 |
| h_sib_neg_share | 1.41 |
| e_fx_share | 1.41 |
| b_below_0 | 1.36 |
| f_ds_r | 1.29 |
| f_fc_r | 1.27 |
| b_neg_episodes | 1.26 |
| b_bal_vol | 1.25 |
| b_d_runway | 1.20 |
| f_has_factoring | 1.18 |
| e_dpo_proxy | 1.17 |
| a_pending_share | 1.10 |
| e_dso_proxy | 1.07 |
| f_new_facility | — |
| g_has_saving | — |

## 4. Company clustering

Input is a **small standardised set of company medians**, not the full 100-d panel. Raw level amounts and SIZE-flagged columns are excluded so the partition is a chance to find operating types rather than 'big vs small'.

Features used: `a_io_ratio`, `a_growth_3`, `b_runway`, `b_bal_vol`, `c_zero_in_share_6`, `c_gap_sd`, `d_cust_hhi`, `d_tx_cp_share`, `e_ap_overdue_30`, `e_ar_overdue_30`, `f_ds_r`, `f_fc_r`.

Cluster sizes: C0=156, C1=856, C2=195, C3=7

Crosstab vs log-inflow tertile (T1 = smallest third of company-median `a_op_in`):

| cluster | T1_small | T2_mid | T3_large |
| --- | --- | --- | --- |
| C0 | 152 | 2 | 2 |
| C1 | 164 | 339 | 353 |
| C2 | 88 | 60 | 47 |
| C3 | 1 | 3 | 3 |

C0 is 97.4% T1_small (n=156) — a size pocket, not a type. C3=7 is too small to model. If groups with ≥ 3 members are mostly mono-cluster, the types are group membership (the holdout is already group-aware, so a second group-cluster model would double-count that cut).

## 5. How to use this in models

- Always put **one** size control in the GBM: `log1p(a_in3)`. Do not also keep `a_op_in` / `a_in6` / `a_in12` / `a_out*` / `a_net`. Company-median VIF of raw euro levels is huge even when panel `|ρ|` is below 0.8 (`a_net = a_op_in − a_op_out`).
- Prefer shares, ratios, overdue-30, runway, gap-sd, HHI over open-amount / issued-amount levels.
- Change features (`a_growth_*`, `b_d_runway`) are allowed to have low lag-1 autocorr — that is the point; do not drop them for persistence alone. Javier already saw level persistence and change mean-reversion.
- Family E DSO/DPO means are unusable (extreme invoices / tiny issued). If `e_dso_proxy` / `e_dpo_proxy` survive as representatives, winsorise at 24 months in the model layer; this report does not fit that clip on data.
- Do not fit anything that produced this report on the 72 holdout companies.

## 6. Plots

- `feature_corr_heatmap.png`
- `feature_dendrogram.png`
- `company_clusters_pca.png`
