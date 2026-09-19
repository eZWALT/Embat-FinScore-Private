# Wave 4 — q6_quoted (end note)

Agent `c0ddcae1`. Files: `analysis/evaluate/q6_quoted.py`, `analysis/outputs/q6_quoted.md`, `analysis/outputs/q6_quoted_lag_share.png`, registry appends. No parquet rewrite. No GBM. No product/. No new Y. Histogram not redone — quote `trail_length.md`.

## What was measured

Quoted leads vs trail buckets (so-far, then company-total). Train rates + group-fold single AUROC (seed 20260918). Holdout 72 coverage only.

## Coverage (train labeled)

- Y3 C/A lag1 **100%** on short so-far (Y3 starts at so-far=2). Lag3 **84.7%** (570/3,723 missing = so-far<4). Same for `c_ss_month` / `a_n_tx`.
- Y3 `f_ds_r_lag3` **61.4%** short; **0%** on so-far<6 (rolling-3 at t−3).
- Y7 `e_ar_issued_lag1` **95.2%** short (confirms trail_length). Hole = so-far<2.
- Y7 `f_fc_r_lag3` **55.1%** short; **0%** on so-far<6. Company-short **16.1%**.
- Y4 `d_cust_hhi_lag3` **21.7%** short (confirms 78% missing). Company-short 26.8%.
- `e_ar_issued_lag_cv` **not in store** — skipped, not invented.

## Y3 lag1/lag3 on short

Present, not a source-NaN hole. C/A lag3 is a **company-relative full4** (100% once so-far≥4, including late first-tx). Not Y4 calendar `full6` (HHI still 24.7% nn after so-far≥4). On <12-month *companies* lag3 drops to **36.3%** → CLOSE as a 3-month lead for short books. F lag3 is a different clock (full6).

## Short vs long single AUROC (train group-fold, nn only)

| col | short so-far | n_pos | vs night | long so-far |
| --- | --- | --- | --- | --- |
| `c_n_days_with_tx` | 0.696 | 252 | 0.711 −0.015 | LOW_POWER (16) |
| `c_ss_month` | 0.693 | 252 | 0.690 +0.003 | LOW_POWER (16) |
| `c_n_days_with_tx_lag1` | 0.684 | 252 | 0.711 −0.027 | LOW_POWER |
| `c_n_days_with_tx_lag3` | 0.691 | 205 | 0.711 −0.020 | LOW_POWER |
| `e_ar_issued_lag1` | 0.626 | 1260 | 0.630 −0.004 | 0.646 |
| `d_cust_hhi_lag3` | LOW_POWER | 42 | 0.605 | LOW_POWER (18) |

days_lag1 on so-far≥4 short = **0.701** (n_pos=205). Company-short issued_lag1 = **0.722** (+0.092 vs night); early months of long books = 0.601.

## Honest source before t (train labeled)

Y3 bank ≥6: 49.2% of short labels (p50=5). Y7 invoice ≥6: 60.1% short (p50=7) / 60.8% given lag. Y4 ERP-customer ≥6: 19.1% short; 67.9% given HHI_lag3.

## KEEP / CLOSE

- **KEEP Q6:** `e_ar_issued_lag1` (present 95%/86% so-far/company-short; CV 0.626 vs 0.630). `c_n_days_with_tx_lag1` (100% both clocks; CV 0.684 vs 0.711, band edge).
- **SIGNAL not Q6:** contemporaneous `c_n_days_with_tx` 0.696 / `c_ss_month` 0.693 on short.
- **CLOSE:** Y3 days_lag3 (missing on company-short half). Y3 `f_ds_r_lag3` / Y7 `f_fc_r_lag3` (full6). Credit-note lag1 (CV 0.542). Y4 HHI_lag3 (21.7%, LOW_POWER 42 pos).

## Hidden 72 (one sentence)

Only the **1-month** claims can be said: issued_lag1 is defined on 362/391 holdout Y7 labels; Y4 HHI_lag3 is 21/135 and the three 24-month books are not a bench (trail_length 4.2% =24). Do not transfer 0.630 — coverage yes, mix no.

## What failed / next

First pass wrote holdout n into the “train” so-far tables (`fmt_cov_table` ignored split). Fixed. Registry skip marker missed `x_families` and duplicated rows until the last run. Next (not this lane): do not invent issued-lag CV; do not revive HHI_lag3 as a hidden-test Q6.
