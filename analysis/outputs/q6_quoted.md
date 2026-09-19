# Quoted Q6 leads vs trail — coverage and single-feature AUROC

Generated `2026-09-19T02:00:42+02:00` by agent `c0ddcae1`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No new Y.

Quote `trail_length.md` (do not redo the histogram): `created_at` 73.6% is the **connection** clock; late first tx 64.2% is the **bank trail**. Train months-on-book: <12 28.8%, ≥18 58.8%, =24 35.8%. Holdout =24 **4.2%**. Y4 `d_cust_hhi_lag3` missing on 78% of short labeled rows. Y7 `e_ar_issued_lag1` present on 95% short; honest≥6 = 47%. **PARK** `created_at` as a health Y.

## Headline

Y3 short lag1 nn 100.0% / lag3 nn 84.7% (so-far<4 = 570 of 3,723); Y7 issued_lag1 nn 95.2%; Y4 HHI_lag3 short present 21.7%. Short singles: days 0.696 vs 0.711; issued_lag1 0.626 vs 0.630; HHI_lag3 LOW_POWER vs 0.605. Survivors: y3_recover `c_n_days_with_tx_lag1`, y7_top1_lost `e_ar_issued_lag1`.

### Quoted engines (unchanged)

- Y3 15-col shallow-A **0.752**: stems + lags 1,3 of `c_ss_month`, `c_salary_month`, `a_n_tx`, `f_ds_r`, `c_n_days_with_tx`. Never B. Never `a_op_in`.
- Y7 TURNOVER **0.720** / n_x=5: issued lag1 + issued-lag CV + credit-note ±lag1 + `f_fc_r_lag3`. No DSO.
- Y4 single `d_cust_hhi_lag3` **0.605** (monopoly tail, not a tree).

`e_ar_issued_lag_cv` is **not** in `monthly.parquet` (it is derived inside `gbm_y7_core.py`). Skipped. Do not invent it here.

## Pass 1 — lead non-null share by months-so-far (train labeled)

Clock is **months-on-book so far** at the company-month (Q6-honest). A 24-month company is still short in its first months. Holdout coverage is a later table.

### Y3 `y3_recover_cash_6m`

| bucket | n_labeled | c_n_days_with_tx | c_n_days_with_tx_lag1 | c_n_days_with_tx_lag3 | c_ss_month | c_ss_month_lag1 | c_ss_month_lag3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <6 | 1,436 | 1,436 (100.0%) | 1,436 (100.0%) | 866 (60.3%) | 1,436 (100.0%) | 1,436 (100.0%) | 866 (60.3%) |
| 6-11 | 2,287 | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) |
| 12-17 | 1,712 | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) |
| 18-23 | 213 | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) |
| 24 | 0 | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) |
| short_<12 | 3,723 | 3,723 (100.0%) | 3,723 (100.0%) | 3,153 (84.7%) | 3,723 (100.0%) | 3,723 (100.0%) | 3,153 (84.7%) |
| long_>=18 | 213 | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) |
| all | 5,648 | 5,648 (100.0%) | 5,648 (100.0%) | 5,078 (89.9%) | 5,648 (100.0%) | 5,648 (100.0%) | 5,078 (89.9%) |


Full 15-col nn shares (train labeled, so-far):

| bucket | n_labeled | c_ss_month | c_ss_month_lag1 | c_ss_month_lag3 | c_salary_month | c_salary_month_lag1 | c_salary_month_lag3 | a_n_tx | a_n_tx_lag1 | a_n_tx_lag3 | f_ds_r | f_ds_r_lag1 | f_ds_r_lag3 | c_n_days_with_tx | c_n_days_with_tx_lag1 | c_n_days_with_tx_lag3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <6 | 1,436 | 1,436 (100.0%) | 1,436 (100.0%) | 866 (60.3%) | 1,436 (100.0%) | 1,436 (100.0%) | 866 (60.3%) | 1,436 (100.0%) | 1,436 (100.0%) | 866 (60.3%) | 1,316 (91.6%) | 866 (60.3%) | 0 (0.0%) | 1,436 (100.0%) | 1,436 (100.0%) | 866 (60.3%) |
| 6-11 | 2,287 | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) |
| 12-17 | 1,712 | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) |
| 18-23 | 213 | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) |
| 24 | 0 | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) |
| short_<12 | 3,723 | 3,723 (100.0%) | 3,723 (100.0%) | 3,153 (84.7%) | 3,723 (100.0%) | 3,723 (100.0%) | 3,153 (84.7%) | 3,723 (100.0%) | 3,723 (100.0%) | 3,153 (84.7%) | 3,603 (96.8%) | 3,153 (84.7%) | 2,287 (61.4%) | 3,723 (100.0%) | 3,723 (100.0%) | 3,153 (84.7%) |
| long_>=18 | 213 | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) |
| all | 5,648 | 5,648 (100.0%) | 5,648 (100.0%) | 5,078 (89.9%) | 5,648 (100.0%) | 5,648 (100.0%) | 5,078 (89.9%) | 5,648 (100.0%) | 5,648 (100.0%) | 5,078 (89.9%) | 5,528 (97.9%) | 5,078 (89.9%) | 4,212 (74.6%) | 5,648 (100.0%) | 5,648 (100.0%) | 5,078 (89.9%) |


### Y7 `y7_top1_lost`

| bucket | n_labeled | e_ar_issued_lag1 | e_credit_note_ratio | e_credit_note_ratio_lag1 | f_fc_r_lag3 |
| --- | --- | --- | --- | --- | --- |
| <6 | 1,983 | 1,772 (89.4%) | 1,943 (98.0%) | 1,701 (85.8%) | 0 (0.0%) |
| 6-11 | 2,438 | 2,438 (100.0%) | 2,380 (97.6%) | 2,385 (97.8%) | 2,438 (100.0%) |
| 12-17 | 2,127 | 2,127 (100.0%) | 2,079 (97.7%) | 2,077 (97.6%) | 2,127 (100.0%) |
| 18-23 | 916 | 916 (100.0%) | 906 (98.9%) | 908 (99.1%) | 916 (100.0%) |
| 24 | 0 | 0 (—) | 0 (—) | 0 (—) | 0 (—) |
| short_<12 | 4,421 | 4,210 (95.2%) | 4,323 (97.8%) | 4,086 (92.4%) | 2,438 (55.1%) |
| long_>=18 | 916 | 916 (100.0%) | 906 (98.9%) | 908 (99.1%) | 916 (100.0%) |
| all | 7,464 | 7,253 (97.2%) | 7,308 (97.9%) | 7,071 (94.7%) | 5,481 (73.4%) |


### Y4 `y4_ds_r_double`

| bucket | n_labeled | d_cust_hhi_lag3 |
| --- | --- | --- |
| <6 | 485 | 42 (8.7%) |
| 6-11 | 849 | 248 (29.2%) |
| 12-17 | 739 | 400 (54.1%) |
| 18-23 | 297 | 158 (53.2%) |
| 24 | 0 | 0 (—) |
| short_<12 | 1,334 | 290 (21.7%) |
| long_>=18 | 297 | 158 (53.2%) |
| all | 2,370 | 848 (35.8%) |


Plot: `q6_quoted_lag_share.png`.

## Pass 2 — Y3 lag1 / lag3 missing on short vs Y7 issued_lag1

Y3 short labeled n=3,723 (so-far<4 = 570; so-far≥4 = 3,153). `c_n_days_with_tx` lag1 nn **100.0%**, lag3 nn **84.7%**. `c_ss_month` lag1 100.0%, lag3 84.7%. Y7 short `e_ar_issued_lag1` nn **95.2%** (trail_length 95.2% — confirm). C/A stems (`c_*`, `a_n_tx`) are 0-filled on the grid, so those lag holes are the panel shift (so-far < k+1). `f_ds_r` is **not** — it needs a rolling 3, so lag3 is empty on all so-far<6 (pass 9). C/A lag3 is a company-relative **full4**; F lag3 is a company-relative **full6**, still not Y4's calendar `full6`.

| y | col | slice | n_lab | missing | shift<k+1 | calendar | source/residual | nn | history |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover | c_n_days_with_tx_lag1 | short | 3,723 | 0 (0.0%) | 0 | 0 (abs) | 0 | 3,723/3,723 |
| y3_recover | c_n_days_with_tx_lag3 | short | 3,723 | 570 (15.3%) | 570 | 256 (abs) | 0 | 3,153/3,153 |
| y3_recover | c_ss_month_lag1 | short | 3,723 | 0 (0.0%) | 0 | 0 (abs) | 0 | 3,723/3,723 |
| y3_recover | c_ss_month_lag3 | short | 3,723 | 570 (15.3%) | 570 | 256 (abs) | 0 | 3,153/3,153 |
| y3_recover | a_n_tx_lag3 | short | 3,723 | 570 (15.3%) | 570 | 256 (abs) | 0 | 3,153/3,153 |
| y3_recover | f_ds_r_lag3 | short | 3,723 | 1,436 (38.6%) | 570 | 256 (abs) | 866 | 2,287/3,153 |
| y3_recover | c_n_days_with_tx_lag3 | all | 5,648 | 570 (10.1%) | 570 | 256 (abs) | 0 | 5,078/5,078 |
| y7_top1_lost | e_ar_issued_lag1 | short | 4,421 | 211 (4.8%) | 211 | 0 (abs) | 0 | 4,210/4,210 |
| y7_top1_lost | e_ar_issued_lag1 | all | 7,464 | 211 (2.8%) | 211 | 0 (abs) | 0 | 7,253/7,253 |
| y4_ds_r_double | d_cust_hhi_lag3 | short | 1,334 | 1,044 (78.3%) | 162 | 405 | 477 / 0 | — |
| y4_ds_r_double | d_cust_hhi_lag3 | all | 2,370 | 1,522 (64.2%) | 162 | 405 | 955 / 0 | — |


## Pass 3 — single-feature train group-fold AUROC (short vs long so-far)

Only rows where the quoted column is non-null. Sign from the train side of each fold. Seed 20260918. If a slice has <50 positives, **LOW_POWER** — do not quote AUROC as a keep. Night quotes: `c_n_days_with_tx` 0.711, `c_ss_month` 0.690 (univ), `e_ar_issued_lag1` 0.630, `d_cust_hhi_lag3` 0.605. KEEP band ±0.03 vs night on the **short** slice, and present on the short half.

| y | col | slice | n_nn | n_pos | present | CV | sd | sign | night | Δ night | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover | c_n_days_with_tx | all | 5,648 | 402 | 100.0% | 0.711 | 0.031 | -1 | 0.711 | +0.000 | quote-only |
| y3_recover | c_n_days_with_tx | short_<12_sofar | 3,723 | 252 | 100.0% | 0.696 | 0.067 | -1 | 0.711 | -0.015 | KEEP |
| y3_recover | c_n_days_with_tx | long_>=18_sofar | 213 | 16 | 100.0% | LOW_POWER | — | — | 0.711 | — | LOW_POWER |
| y3_recover | c_ss_month | all | 5,648 | 402 | 100.0% | 0.693 | 0.031 | -1 | 0.690 | +0.003 | quote-only |
| y3_recover | c_ss_month | short_<12_sofar | 3,723 | 252 | 100.0% | 0.693 | 0.064 | -1 | 0.690 | +0.003 | KEEP |
| y3_recover | c_ss_month | long_>=18_sofar | 213 | 16 | 100.0% | LOW_POWER | — | — | 0.690 | — | LOW_POWER |
| y7_top1_lost | e_ar_issued_lag1 | all | 7,253 | 2,072 | 97.2% | 0.630 | 0.031 | -1 | 0.630 | -0.000 | quote-only |
| y7_top1_lost | e_ar_issued_lag1 | short_<12_sofar | 4,210 | 1,260 | 95.2% | 0.626 | 0.068 | -1 | 0.630 | -0.004 | KEEP |
| y7_top1_lost | e_ar_issued_lag1 | long_>=18_sofar | 916 | 209 | 100.0% | 0.646 | 0.077 | -1 | 0.630 | +0.016 | quote-only |
| y4_ds_r_double | d_cust_hhi_lag3 | all | 848 | 116 | 35.8% | 0.605 | 0.044 | 1 | 0.605 | -0.000 | quote-only |
| y4_ds_r_double | d_cust_hhi_lag3 | short_<12_sofar | 290 | 42 | 21.7% | LOW_POWER | — | — | 0.605 | — | LOW_POWER |
| y4_ds_r_double | d_cust_hhi_lag3 | long_>=18_sofar | 158 | 18 | 53.2% | LOW_POWER | — | — | 0.605 | — | LOW_POWER |


Lag-as-single (same rule; not a tree):

| y | col | slice | n_nn | n_pos | present | CV | Δ vs stem-night |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover | c_n_days_with_tx_lag1 | all | 5,648 | 402 | 100.0% | 0.684 | -0.027 |
| y3_recover | c_n_days_with_tx_lag1 | short_<12_sofar | 3,723 | 252 | 100.0% | 0.684 | -0.027 |
| y3_recover | c_n_days_with_tx_lag1 | long_>=18_sofar | 213 | 16 | 100.0% | LOW_POWER | — |
| y3_recover | c_n_days_with_tx_lag3 | all | 5,078 | 355 | 89.9% | 0.666 | -0.045 |
| y3_recover | c_n_days_with_tx_lag3 | short_<12_sofar | 3,153 | 205 | 84.7% | 0.691 | -0.020 |
| y3_recover | c_n_days_with_tx_lag3 | long_>=18_sofar | 213 | 16 | 100.0% | LOW_POWER | — |
| y3_recover | c_ss_month_lag1 | all | 5,648 | 402 | 100.0% | 0.679 | -0.011 |
| y3_recover | c_ss_month_lag1 | short_<12_sofar | 3,723 | 252 | 100.0% | 0.686 | -0.004 |
| y3_recover | c_ss_month_lag1 | long_>=18_sofar | 213 | 16 | 100.0% | LOW_POWER | — |
| y3_recover | c_ss_month_lag3 | all | 5,078 | 355 | 89.9% | 0.665 | -0.025 |
| y3_recover | c_ss_month_lag3 | short_<12_sofar | 3,153 | 205 | 84.7% | 0.676 | -0.014 |
| y3_recover | c_ss_month_lag3 | long_>=18_sofar | 213 | 16 | 100.0% | LOW_POWER | — |
| y3_recover | f_ds_r_lag3 | all | 4,212 | 313 | 74.6% | 0.604 | -0.107 |
| y3_recover | f_ds_r_lag3 | short_<12_sofar | 2,287 | 163 | 61.4% | 0.604 | -0.107 |
| y3_recover | f_ds_r_lag3 | long_>=18_sofar | 213 | 16 | 100.0% | LOW_POWER | — |
| y7_top1_lost | e_credit_note_ratio_lag1 | all | 7,071 | 1,969 | 94.7% | 0.548 | -0.082 |
| y7_top1_lost | e_credit_note_ratio_lag1 | short_<12_sofar | 4,086 | 1,198 | 92.4% | 0.542 | -0.088 |
| y7_top1_lost | e_credit_note_ratio_lag1 | long_>=18_sofar | 908 | 203 | 99.1% | 0.555 | -0.075 |
| y7_top1_lost | f_fc_r_lag3 | all | 5,481 | 1,496 | 73.4% | 0.571 | -0.059 |
| y7_top1_lost | f_fc_r_lag3 | short_<12_sofar | 2,438 | 684 | 55.1% | 0.553 | -0.077 |
| y7_top1_lost | f_fc_r_lag3 | long_>=18_sofar | 916 | 209 | 100.0% | 0.640 | +0.010 |


## Pass 4 — honest source length before t (train labeled)

Share of labeled rows with ≥3 / ≥6 months of *that source* strictly before t. Bank tx months for Y3, any book-invoice months for Y7, AR-customer invoice months for Y4. This is not `months_so_far − k` (quoted from trail_length: Y7 short honest≥6 = 47% given issued_lag1; Y4 short honest≥6 = 66% given HHI_lag3).

| y | slice | n_lab | source | p50 before t | ≥3 | ≥6 | lead nn | ge6 given nn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover | all | 5,648 | n_bank_before | 8.0 | 89.5% | 66.5% | 5,078 | 73.9% |
| y3_recover | short_<12 | 3,723 | n_bank_before | 5.0 | 84.0% | 49.2% | 3,153 | 58.1% |
| y3_recover | long_>=18 | 213 | n_bank_before | 17.0 | 100.0% | 100.0% | 213 | 100.0% |
| y3_recover | short_<12_company | 146 | n_bank_before | 2.0 | 36.3% | 0.0% | 53 | 0.0% |
| y3_recover | long_>=18_company | 5,138 | n_bank_before | 9.0 | 91.6% | 70.6% | 4,722 | 76.8% |
| y7_top1_lost | all | 7,464 | n_inv_before | 10.0 | 92.2% | 74.4% | 7,253 | 75.2% |
| y7_top1_lost | short_<12 | 4,421 | n_inv_before | 7.0 | 88.0% | 60.1% | 4,210 | 60.8% |
| y7_top1_lost | long_>=18 | 916 | n_inv_before | 18.0 | 99.2% | 97.5% | 916 | 97.5% |
| y7_top1_lost | short_<12_company | 774 | n_inv_before | 9.0 | 89.5% | 71.2% | 666 | 71.5% |
| y7_top1_lost | long_>=18_company | 6,230 | n_inv_before | 10.0 | 92.7% | 75.1% | 6,154 | 75.9% |
| y4_ds_r_double | all | 2,370 | n_cust_before | 1.0 | 43.0% | 31.1% | 848 | 78.9% |
| y4_ds_r_double | short_<12 | 1,334 | n_cust_before | 0.0 | 33.3% | 19.1% | 290 | 67.9% |
| y4_ds_r_double | long_>=18 | 297 | n_cust_before | 5.0 | 57.6% | 48.5% | 158 | 87.3% |
| y4_ds_r_double | short_<12_company | 142 | n_cust_before | 0.0 | 31.0% | 29.6% | 38 | 81.6% |
| y4_ds_r_double | long_>=18_company | 2,059 | n_cust_before | 2.0 | 45.7% | 32.6% | 776 | 79.5% |


## Pass 5 — holdout coverage only (no AUROC claim)

On the hidden 72 (train quote only; =24 books = 3 / 72 = 4.2% per trail_length) Y4 `d_cust_hhi_lag3` is defined on 21/135 labeled rows (15.6%); Y7 `e_ar_issued_lag1` on 362/391 (92.6%); Y3 days_lag3 on 209/235 (88.9%). A 3-month HHI lead cannot be said on the hidden 72 (almost no 24-month books, and the lag is almost undefined). A 1-month issued lead *is* defined. Y3 lag3 is defined wherever so-far≥4 — that is most holdout Y3 labels, not a 24-month privilege. Do not transfer the Y7 0.630 night number onto the hidden 72: company-short issued_lag1 CV is 0.722 (+0.092); early months of long books are 0.601. Coverage transfers; the AUROC mix may not.

| y | lead | slice | n_labeled | n_nn | share_nn | n_pos |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover | c_n_days_with_tx_lag3 | all | 235 | 209 | 88.9% | 14 |
| y3_recover | c_n_days_with_tx_lag3 | short_<12 | 188 | 162 | 86.2% | 14 |
| y3_recover | c_n_days_with_tx_lag3 | eq24_book | 33 | 30 | 90.9% | 0 |
| y3_recover | c_n_days_with_tx_lag1 | all | 235 | 235 | 100.0% | 14 |
| y3_recover | c_n_days_with_tx_lag1 | short_<12 | 188 | 188 | 100.0% | 14 |
| y3_recover | c_n_days_with_tx_lag1 | eq24_book | 33 | 33 | 100.0% | 0 |
| y7_top1_lost | e_ar_issued_lag1 | all | 391 | 362 | 92.6% | 122 |
| y7_top1_lost | e_ar_issued_lag1 | short_<12 | 319 | 290 | 90.9% | 113 |
| y7_top1_lost | e_ar_issued_lag1 | eq24_book | 57 | 57 | 100.0% | 11 |
| y4_ds_r_double | d_cust_hhi_lag3 | all | 135 | 21 | 15.6% | 16 |
| y4_ds_r_double | d_cust_hhi_lag3 | short_<12 | 88 | 18 | 20.5% | 11 |
| y4_ds_r_double | d_cust_hhi_lag3 | eq24_book | 2 | 2 | 100.0% | 0 |


## Pass 6 — same coverage on company-total trail (not so-far)

Robustness clock: the company's official grid span. Prefer this for *who* has a long book; so-far remains the Q6-honest clock for *when* a lead is defined. Y3 so-far ≥18 is a Feb-2026 sliver of 2024-09 starters (trail_length pass 19) — do not read so-far-long AUROC as a long-book world.

### Y3 company-total

| bucket | n_labeled | c_n_days_with_tx | c_n_days_with_tx_lag1 | c_n_days_with_tx_lag3 | c_ss_month | c_ss_month_lag1 | c_ss_month_lag3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <6 | 0 | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) | 0 (—) |
| 6-11 | 146 | 146 (100.0%) | 146 (100.0%) | 53 (36.3%) | 146 (100.0%) | 146 (100.0%) | 53 (36.3%) |
| 12-17 | 364 | 364 (100.0%) | 364 (100.0%) | 303 (83.2%) | 364 (100.0%) | 364 (100.0%) | 303 (83.2%) |
| 18-23 | 1,677 | 1,677 (100.0%) | 1,677 (100.0%) | 1,511 (90.1%) | 1,677 (100.0%) | 1,677 (100.0%) | 1,511 (90.1%) |
| 24 | 3,461 | 3,461 (100.0%) | 3,461 (100.0%) | 3,211 (92.8%) | 3,461 (100.0%) | 3,461 (100.0%) | 3,211 (92.8%) |
| short_<12 | 146 | 146 (100.0%) | 146 (100.0%) | 53 (36.3%) | 146 (100.0%) | 146 (100.0%) | 53 (36.3%) |
| long_>=18 | 5,138 | 5,138 (100.0%) | 5,138 (100.0%) | 4,722 (91.9%) | 5,138 (100.0%) | 5,138 (100.0%) | 4,722 (91.9%) |
| all | 5,648 | 5,648 (100.0%) | 5,648 (100.0%) | 5,078 (89.9%) | 5,648 (100.0%) | 5,648 (100.0%) | 5,078 (89.9%) |


### Y7 company-total

| bucket | n_labeled | e_ar_issued_lag1 | e_credit_note_ratio | e_credit_note_ratio_lag1 | f_fc_r_lag3 |
| --- | --- | --- | --- | --- | --- |
| <6 | 2 | 1 (50.0%) | 2 (100.0%) | 1 (50.0%) | 0 (0.0%) |
| 6-11 | 772 | 665 (86.1%) | 763 (98.8%) | 640 (82.9%) | 125 (16.2%) |
| 12-17 | 460 | 433 (94.1%) | 452 (98.3%) | 417 (90.7%) | 269 (58.5%) |
| 18-23 | 2,353 | 2,277 (96.8%) | 2,285 (97.1%) | 2,198 (93.4%) | 1,830 (77.8%) |
| 24 | 3,877 | 3,877 (100.0%) | 3,806 (98.2%) | 3,815 (98.4%) | 3,257 (84.0%) |
| short_<12 | 774 | 666 (86.0%) | 765 (98.8%) | 641 (82.8%) | 125 (16.1%) |
| long_>=18 | 6,230 | 6,154 (98.8%) | 6,091 (97.8%) | 6,013 (96.5%) | 5,087 (81.7%) |
| all | 7,464 | 7,253 (97.2%) | 7,308 (97.9%) | 7,071 (94.7%) | 5,481 (73.4%) |


### Y4 company-total

| bucket | n_labeled | d_cust_hhi_lag3 |
| --- | --- | --- |
| <6 | 0 | 0 (—) |
| 6-11 | 142 | 38 (26.8%) |
| 12-17 | 169 | 34 (20.1%) |
| 18-23 | 669 | 264 (39.5%) |
| 24 | 1,390 | 512 (36.8%) |
| short_<12 | 142 | 38 (26.8%) |
| long_>=18 | 2,059 | 776 (37.7%) |
| all | 2,370 | 848 (35.8%) |


Single-feature AUROC on company-total short vs long (train labeled, feature non-null):

| y | col | slice | n_nn | n_pos | present | CV | Δ night |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover | c_n_days_with_tx | short_<12_company | 146 | 13 | 100.0% | LOW_POWER | — |
| y3_recover | c_n_days_with_tx | long_>=18_company | 5,138 | 355 | 100.0% | 0.714 | +0.003 |
| y3_recover | c_ss_month | short_<12_company | 146 | 13 | 100.0% | LOW_POWER | — |
| y3_recover | c_ss_month | long_>=18_company | 5,138 | 355 | 100.0% | 0.690 | -0.000 |
| y7_top1_lost | e_ar_issued_lag1 | short_<12_company | 666 | 140 | 86.0% | 0.722 | +0.092 |
| y7_top1_lost | e_ar_issued_lag1 | long_>=18_company | 6,154 | 1,813 | 98.8% | 0.616 | -0.014 |
| y4_ds_r_double | d_cust_hhi_lag3 | short_<12_company | 38 | 4 | 26.8% | LOW_POWER | — |
| y4_ds_r_double | d_cust_hhi_lag3 | long_>=18_company | 776 | 106 | 37.7% | 0.628 | +0.023 |


## Pass 7 — is Y3 lag3 a calendar full4 like Y4 full6?

Y3 `*_lag3` is present on **100.0%** of short labeled rows with so-far≥4, including late first-tx books. Missing lag3 is almost only so-far<4 (panel shift). Y4 `d_cust_hhi_lag3` is still missing on short even after so-far≥4 (nn share 24.7%) — calendar `full6` at t−3 plus no ERP HHI, same as trail_length pass 10. Y3 lag3 is **not** a 2024-12 calendar gate like Y4 `full6`.

| test | n | n_nn | share |
| --- | --- | --- | --- |
| Y3 short ∧ so-far≥4 → days_lag3 nn | 3153 | 3153 | 100.0% |
| Y3 late-start ∧ so-far≥4 → days_lag3 nn | 1867 | 1867 | 100.0% |
| Y3 period<2024-12 (abs calendar) among short miss lag3 | 570 | 256 | 44.9% |
| Y4 short ∧ so-far≥4 → HHI_lag3 nn | 1172 | 290 | 24.7% |
| Y4 short ∧ period≥2025-05 (full6 at t−3) → HHI_lag3 nn | 833 | 290 | 34.8% |


## Pass 8 — company-short CLOSE, f_ds_r residual, so-far 2–3 vs ≥4

On <12-month *companies*, Y3 days_lag3 is present on only **36.3%** of labeled rows (so-far-short was 84.7%). Y3 days_lag1 stays **100.0%**. Y7 issued_lag1 company-short nn **86.0%**. Y7 `f_fc_r_lag3` company-short nn **16.1%** — CLOSE as a 3-month TURNOVER lead on short books. Y3 `f_ds_r` stem nn on short 96.8%; lag3 nn given so-far≥4 72.5% (stem-null share on those rows 0.0%). Y7 short `f_fc_r` stem 89.5%; lag3 holes are 946 shift + 1,037 residual.

| cut | n | n_nn | share |
| --- | --- | --- | --- |
| Y3 company-short days_lag3 nn | 146 | 53 | 36.3% |
| Y3 company-short days_lag1 nn | 146 | 146 | 100.0% |
| Y7 company-short issued_lag1 nn | 774 | 666 | 86.0% |
| Y7 company-short f_fc_r_lag3 nn | 774 | 125 | 16.1% |
| Y4 company-short HHI_lag3 nn | 142 | 38 | 26.8% |
| Y3 short f_ds_r stem nn | 3723 | 3603 | 96.8% |
| Y3 short so-far≥4 f_ds_r_lag3 nn | 3153 | 2287 | 72.5% |
| Y3 short f_ds_r_lag3 miss after so-far≥4 | 3723 | 866 | 23.3% |
| Y7 short f_fc_r_lag3 miss = shift so-far<4 | 1983 | 946 | 47.7% |
| Y7 short f_fc_r_lag3 miss residual so-far≥4 | 1983 | 1037 | 52.3% |


Extra singles (train group-fold, feature non-null):

| y | col | slice | n_nn | n_pos | present | CV | Δ night |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover | c_n_days_with_tx | sofar_2_3 | 570 | 47 | 100.0% | LOW_POWER | — |
| y3_recover | c_n_days_with_tx | sofar_ge4_short | 3,153 | 205 | 100.0% | 0.715 | +0.004 |
| y3_recover | c_n_days_with_tx_lag1 | sofar_2_3 | 570 | 47 | 100.0% | LOW_POWER | — |
| y3_recover | c_n_days_with_tx_lag1 | sofar_ge4_short | 3,153 | 205 | 100.0% | 0.701 | -0.010 |
| y3_recover | c_n_days_with_tx_lag3 | sofar_ge4_short | 3,153 | 205 | 100.0% | 0.691 | -0.020 |
| y3_recover | c_n_days_with_tx | mid_12_17_sofar | 1,712 | 134 | 100.0% | 0.678 | -0.033 |
| y7_top1_lost | e_ar_issued_lag1 | mid_12_17_sofar | 2,127 | 603 | 100.0% | 0.626 | -0.004 |
| y7_top1_lost | e_ar_issued_lag1 | sofar_short_and_co_long | 3,142 | 1,010 | 97.6% | 0.601 | -0.029 |
| y7_top1_lost | e_ar_issued_lag1 | sofar_short_and_co_short | 666 | 140 | 86.0% | 0.722 | +0.092 |
| y4_ds_r_double | d_cust_hhi_lag3 | mid_12_17_sofar | 400 | 56 | 54.1% | 0.700 | +0.095 |
| y7_top1_lost | f_fc_r_lag3 | sofar_ge4_short | 2,438 | 684 | 70.2% | 0.553 | -0.077 |


## Pass 9 — `f_ds_r` / `f_fc_r` lag3 is a rolling-3 full6, not C/A full4

Family F ratios need `in3`/`ds3` (rolling 3). Y3 `f_ds_r` nn is 0.0% on so-far<3 and 100.0% on so-far≥3. Therefore `f_ds_r_lag3` nn is **0.0%** on so-far<6 and 100.0% on so-far≥6. Y7 `f_fc_r_lag3` is the same clock: 0.0% on so-far<6, 100.0% on so-far≥6. The quoted 15-col / TURNOVER 3-month F leads are a **company-relative full6**, not the C/A full4. Still not Y4 calendar `full6`.

| y | so-far | n_lab | stem | stem nn | lag3 | lag3 nn |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover | 1 | 0 | f_ds_r | — | f_ds_r_lag3 | — |
| y3_recover | 2 | 120 | f_ds_r | 0.0% | f_ds_r_lag3 | 0.0% |
| y3_recover | 3 | 450 | f_ds_r | 100.0% | f_ds_r_lag3 | 0.0% |
| y3_recover | 4 | 433 | f_ds_r | 100.0% | f_ds_r_lag3 | 0.0% |
| y3_recover | 5 | 433 | f_ds_r | 100.0% | f_ds_r_lag3 | 0.0% |
| y3_recover | 6 | 408 | f_ds_r | 100.0% | f_ds_r_lag3 | 100.0% |
| y3_recover | 7 | 396 | f_ds_r | 100.0% | f_ds_r_lag3 | 100.0% |
| y3_recover | 8 | 388 | f_ds_r | 100.0% | f_ds_r_lag3 | 100.0% |
| y7_top1_lost | 1 | 211 | f_fc_r | 0.0% | f_fc_r_lag3 | 0.0% |
| y7_top1_lost | 2 | 253 | f_fc_r | 0.0% | f_fc_r_lag3 | 0.0% |
| y7_top1_lost | 3 | 482 | f_fc_r | 100.0% | f_fc_r_lag3 | 0.0% |
| y7_top1_lost | 4 | 520 | f_fc_r | 100.0% | f_fc_r_lag3 | 0.0% |
| y7_top1_lost | 5 | 517 | f_fc_r | 100.0% | f_fc_r_lag3 | 0.0% |
| y7_top1_lost | 6 | 459 | f_fc_r | 100.0% | f_fc_r_lag3 | 100.0% |
| y7_top1_lost | 7 | 420 | f_fc_r | 100.0% | f_fc_r_lag3 | 100.0% |
| y7_top1_lost | 8 | 409 | f_fc_r | 100.0% | f_fc_r_lag3 | 100.0% |


## KEEP / CLOSE (honest Q6 sentence)

KEEP as an honest Q6 sentence only: y3_recover `c_n_days_with_tx_lag1`; y7_top1_lost `e_ar_issued_lag1`. Contemporaneous Y3 days/ss are SIGNAL (rank on short books) not lead time. CLOSE if missing on the so-far short half *or* the company-short half, AUROC off the night quote by >0.03, or LOW_POWER n_pos<50.

| claim | kind | so-far nn | co-short nn | so-far n_pos | so-far CV | night | Δ | so-far rule | Q6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover `c_n_days_with_tx` | signal | 100.0% | 100.0% | 252 | 0.696 | 0.711 | -0.015 | KEEP | SIGNAL |
| y3_recover `c_ss_month` | signal | 100.0% | 100.0% | 252 | 0.693 | 0.690 | +0.003 | KEEP | SIGNAL |
| y3_recover `c_n_days_with_tx_lag1` | q6 | 100.0% | 100.0% | 252 | 0.684 | 0.711 | -0.027 | KEEP | KEEP |
| y3_recover `c_n_days_with_tx_lag3` | q6 | 84.7% | 36.3% | 205 | 0.691 | 0.711 | -0.020 | KEEP | CLOSE |
| y3_recover `f_ds_r_lag3` | q6 | 61.4% | 0.0% | 163 | 0.604 | 0.711 | -0.107 | CLOSE | CLOSE |
| y7_top1_lost `e_ar_issued_lag1` | q6 | 95.2% | 86.0% | 1260 | 0.626 | 0.630 | -0.004 | KEEP | KEEP |
| y7_top1_lost `e_credit_note_ratio_lag1` | q6 | 92.4% | 82.8% | 1198 | 0.542 | 0.630 | -0.088 | CLOSE | CLOSE |
| y7_top1_lost `f_fc_r_lag3` | q6 | 55.1% | 16.1% | 684 | 0.553 | 0.630 | -0.077 | CLOSE | CLOSE |
| y4_ds_r_double `d_cust_hhi_lag3` | q6 | 21.7% | 26.8% | 42 | LOW_POWER | 0.605 | — | LOW_POWER | CLOSE |


## Six brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Not a health reading. Trail gates the lead, not the label. |
| 2 | Who is improving? | A short book cannot show a 12-month improvement. |
| 3 | Who is turning? | Y4 HHI_lag3 is missing on the short half — no 3-month turn clock there. |
| 4 | Dip vs fall? | Y7 issued_lag1 is present on short; the cap is length, not a hole. |
| 5 | Why did it change? | Contemporaneous Y3 singles can still rank on short books; that is not lead time. |
| 6 | Months earlier? | Y7 issued_lag1 KEEP (present 95.2% short, CV 0.626 vs 0.630); Y4 HHI_lag3 CLOSE (present 21.7% — missing on the short half); Y3 contemporaneous days SIGNAL (signal, not a lead); Y3 days_lag3 CLOSE (so-far nn 84.7%; company-short nn 36.3% — full4 shift, not calendar full6). |

Elapsed 3s. Cuts: so-far coverage, Y3 vs Y7 missingness, short/long singles, source-honest length, holdout coverage, company-total, Y3 full4 vs Y4 full6, company-short CLOSE / f_ds_r / so-far 2–3, F full6, KEEP/CLOSE.

## What failed / next

- `e_ar_issued_lag_cv` not in store — skipped (not invented). TURNOVER coverage uses issued_lag1 + credit-note ±lag1 + f_fc_r_lag3 only.
- Y3 so-far≥18 labeled is a Feb-2026 sliver (trail_length) — long so-far AUROC is LOW_POWER or a calendar mix, not a long-book world. Prefer company-total for who.
- Y4 `d_cust_hhi_lag3` CLOSE as Q6 on short: missing on the short half (21.7%); trail_length 21.7%.
- Y3 days_lag3 CLOSE as Q6 on short *companies* (nn 36.3%) even though so-far-short is 84.7%.
- Y7 `f_fc_r_lag3` CLOSE as a 3-month TURNOVER lead on short companies (nn 16.1%).
- Y3 `f_ds_r_lag3` is 0% on so-far<6 (rolling-3 at t−3). The 15-col is two clocks: C/A full4 vs F full6. CLOSE the F 3-month lead on short books.
