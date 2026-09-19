# Missing transaction `counterparty_id` — leftover after uncat?

Generated `2026-09-19T04:16:17+02:00` by agent `c91e4b2a`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_missing_cp`. Do not merge a Family D column. Do not put this on the 15-col Y3 card. Y7 never uses D. Night Y3 0.762/0.752 and Y5 `d_tx_cp_share` 0.611 quotes unchanged.

`miss_cp_share` = calendar-month share of txs with null/blank `counterparty_id` (in-memory). `d_tx_cp_share` = 6-month *named* fill (store). `miss_mapped_share` = missing-CP among non-uncat txs that month.

## Headline

Train txs miss-CP 90.1% / |amt| 96.4%. Uncat×miss CONFIRM 91.8%; mapped still miss 89.6%. ρ vs uncat 0.131 (not twin); ρ vs d_tx_cp_share -0.947 (TWIN). Y3 miss 0.554 vs size 0.617 (Δ -0.063) vs days 0.711. Leftover after uncat+dtx 0.509. Dark vs 744 0.999 / 0.775. Invoice fill 0.999 vs tx named 0.252. Fold-3 one-group=True. ICC 0.967. X **CLOSE**. Y **PARK**. Q5 **CLOSE**. Q6 **CLOSE**.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Missing-CP is **PARK** as a health Y. Do not invent `y_missing_cp`. |
| 2 | Who is improving? | Not this share. |
| 3 | Who is turning? | A leftover tagging shock would be Q3; a style/fill twin is not. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | **CLOSE** — Bank-book tagging hole is already the Y5 `d_tx_cp_share` sentence (6m named fill). Not a new KEEP-Q5. X is **CLOSE**. CLOSE as X — d_tx_cp_share twin ρ=-0.947; fold-3 / one-group again; leftover after both Y3 0.509 vs size 0.617 (Δ -0.108). |
| 6 | Months earlier? | **CLOSE** — Y3 miss now 0.554 / lag1 0.555 / lag3 0.551. Short so-far<12 0.545 lag1 0.557; long≥18 —. Q6 CLOSE — lag does not hold contemporaneous skill (or short books die). |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| miss_cp_share as Y3 X | **CLOSE** | CLOSE as X — d_tx_cp_share twin ρ=-0.947; fold-3 / one-group again; leftover after both Y3 0.509 vs size 0.617 (Δ -0.108). |
| miss_cp_share as a health Y | **PARK** | do not invent `y_missing_cp` |
| Q5 diagnostic | **CLOSE** | Bank-book tagging hole is already the Y5 `d_tx_cp_share` sentence (6m named fill). Not a new KEEP-Q5. |
| Q6 lag1/lag3 | **CLOSE** | Y3 miss now 0.554 / lag1 0.555 / lag3 0.551. Short so-far<12 0.545 lag1 0.557; long≥18 —. Q6 CLOSE — lag does not hold contemporaneous skill (or short books die). |
| uncat twin (a) | **no** | ρ=0.131 vs `a_uncat_share` |
| d_tx_cp_share twin (b) | **YES — CLOSE** | ρ vs d_tx=-0.947; monthly named vs 6m ρ=0.947 |
| leftover after both (c) | **dies / no KEEP** | Y3 resid 0.509 vs size 0.617 |
| dark 470 hole (d) | **level — not only-470** | invoiced 0.775 vs dark 0.999 |
| Y5 fold-3 again (e) | **YES — CLOSE** | Y5 AR hole (d_tx_cp_share≤0.01 ∩ hi d_n_cust) pos in fold 3: 85.9% of hole positives. High-miss AR months in fold 3: 42.3%. Without fold 3: hole 7.8% vs named 7.0% (survives=False). CLOSE as one-group — same fold-3 that owned the d_tx_cp_share Q5 sentence. |
| Y7 as X | **never** | Y7 models never use D |
| 15-col Y3 card | **not added** | night quote stays 0.762 / 0.752 |
| parquet merge | **not done** | parent decides |

## 1. Prevalence

Train txs missing-CP 90.1% of count / 96.4% of |amt| (2,178,115 / 2,416,107). Company-month share mean 0.855 p50 1.000 cov 95.8% (in-memory; not written). Holdout coverage only: txs 91.1% / |amt| 99.9%; cm defined 97.3%.

| split | n_tx | miss_n | miss_|amt| | cm_defined | cm_mean | cm_p50 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 2,416,107 | 90.1% | 96.4% | 95.8% | 0.855 | 1.000 |
| holdout | 130,719 | 91.1% | 99.9% | 97.3% | 0.881 | 1.000 |


## 2. Uncat × missing-CP 2×2

Uncat txs that miss CP 91.8% of count / 94.0% of |amt| (CONFIRM 91.8%; CONFIRM 94.0% |amt|). Mapped (non-uncat) txs that still miss CP: 89.6% of count / 97.4% of |amt| (1,633,243 / 1,822,584). Mapped still mostly unnamed — missing-CP is not only the uncat token.

| uncat | miss_cp | n | share_n | |amt| | share_|amt| |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 189,341 | 7.8% | 8,564,713,726 | 1.8% |
| 0 | 1 | 1,633,243 | 67.6% | 316,403,386,152 | 68.2% |
| 1 | 0 | 48,651 | 2.0% | 8,297,331,415 | 1.8% |
| 1 | 1 | 544,872 | 22.6% | 130,480,561,653 | 28.1% |


## 3. Spearman twins / SIZE

miss_cp_share vs a_uncat_share ρ=0.131 (not twin). vs d_tx_cp_share ρ=-0.947 / vs 1−d_tx ρ=0.947 (TWIN of fill-rate). vs log1p(a_in3) ρ=-0.029 (not SIZE). vs days ρ=-0.054.

| a | b | ρ | twin_|ρ|≥0.80 | SIZE_|ρ|≥0.50 |
| --- | --- | --- | --- | --- |
| miss_cp_share | a_uncat_share | 0.131 | no | — |
| miss_cp_share | d_tx_cp_share | -0.947 | YES | — |
| miss_cp_share | 1-d_tx_cp_share | 0.947 | YES | — |
| miss_cp_share | log1p(a_in3) | -0.029 | no | — |
| miss_cp_share | c_n_days_with_tx | -0.054 | no | — |
| miss_cp_share | miss_cp_amt | 0.974 | YES | — |
| miss_mapped_share | a_uncat_share | 0.141 | no | — |
| miss_mapped_share | d_tx_cp_share | -0.923 | YES | — |
| named_share | d_tx_cp_share | 0.947 | YES | — |
| miss_cp_amt | a_uncat_share | 0.125 | no | — |
| miss_cp_amt | d_tx_cp_share | -0.926 | YES | — |
| miss_cp_amt | log1p(a_in3) | -0.033 | no | — |


## 4. Single-feature train group-fold AUROC

Y3 miss_cp_share 0.554 vs size 0.617 (Δ -0.063; quote 0.617 CONFIRM) vs days 0.711 (night 0.711 CONFIRM). Y3 d_tx_cp_share replica 0.534. Y2 miss 0.585. Sign from train side of each fold. Never Y7.

Sign from the train side of each fold. Days bar 0.711. Size `log1p(a_in3)` 0.617. Never Y7.

| y | feature | n | n_pos | CV | sd | sign | train | Δsize | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | miss_cp_share | 5,536 | 372 | 0.554 | 0.040 | 1 | 0.556 | -0.063 | 0.517 0.530 0.528 0.585 0.608 |
| y3_recover_cash_6m | miss_cp_amt | 5,536 | 372 | 0.552 | 0.044 | 1 | 0.556 | -0.065 | 0.499 0.542 0.528 0.576 0.613 |
| y3_recover_cash_6m | miss_mapped_share | 5,494 | 365 | 0.566 | 0.051 | 1 | 0.569 | -0.051 | 0.513 0.540 0.534 0.615 0.626 |
| y3_recover_cash_6m | d_tx_cp_share | 5,643 | 402 | 0.534 | 0.055 | -1 | 0.541 | -0.083 | 0.449 0.532 0.534 0.551 0.601 |
| y3_recover_cash_6m | a_uncat_share | 5,536 | 372 | 0.542 | 0.046 | 1 | 0.534 | -0.075 | 0.528 0.542 0.553 0.607 0.478 |
| y3_recover_cash_6m | named_share | 5,536 | 372 | 0.554 | 0.040 | -1 | 0.556 | -0.063 | 0.517 0.530 0.528 0.585 0.608 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 | 0.000 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 | 0.095 | 0.665 0.738 0.700 0.715 0.740 |
| y2_neg_2of3 | miss_cp_share | 16,764 | 1,236 | 0.585 | 0.035 | 1 | 0.572 | 0.033 | 0.620 0.604 0.607 0.553 0.543 |
| y2_neg_2of3 | miss_cp_amt | 16,764 | 1,236 | 0.583 | 0.040 | 1 | 0.568 | 0.031 | 0.617 0.602 0.616 0.547 0.532 |
| y2_neg_2of3 | miss_mapped_share | 16,321 | 1,193 | 0.586 | 0.032 | 1 | 0.576 | 0.034 | 0.618 0.604 0.604 0.555 0.548 |
| y2_neg_2of3 | d_tx_cp_share | 17,271 | 1,269 | 0.590 | 0.034 | -1 | 0.576 | 0.038 | 0.621 0.603 0.616 0.561 0.546 |
| y2_neg_2of3 | a_uncat_share | 16,764 | 1,236 | 0.584 | 0.031 | 1 | 0.576 | 0.033 | 0.575 0.553 0.603 0.628 0.563 |
| y2_neg_2of3 | named_share | 16,764 | 1,236 | 0.585 | 0.035 | -1 | 0.572 | 0.033 | 0.620 0.604 0.607 0.553 0.543 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 | 0.000 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.046 | 1 | 0.577 | 0.019 | 0.623 0.539 0.612 0.565 0.517 |


## 5. Residual after uncat / `d_tx_cp_share`

Y3 leftover after a_uncat_share 0.542; after d_tx_cp_share 0.524; after both 0.509 vs size 0.617 (Δ -0.108). Leftover dies — CLOSE as twin.

| y | feature | n | n_pos | CV | sign | Δsize |
| --- | --- | --- | --- | --- | --- | --- |
| Y3 | resid_uncat | 5,536 | 372 | 0.542 | 1 | -0.075 |
| Y3 | resid_dtx | 5,536 | 372 | 0.524 | 1 | -0.093 |
| Y3 | resid_uncat+dtx | 5,536 | 372 | 0.509 | 1 | -0.108 |
| Y3 | mapped_resid_uncat | 5,494 | 365 | 0.550 | 1 | -0.067 |
| Y3 | mapped_resid_dtx | 5,494 | 365 | 0.534 | 1 | -0.083 |
| Y3 | miss_mapped_share | 5,494 | 365 | 0.566 | 1 | -0.051 |
| Y2 | resid_uncat | 16,764 | 1,236 | 0.544 | 1 | -0.008 |
| Y2 | resid_dtx | 16,764 | 1,236 | 0.517 | 1 | -0.035 |
| Y2 | resid_uncat+dtx | 16,764 | 1,236 | 0.517 | -1 | -0.035 |
| Y2 | mapped_resid_uncat | 16,321 | 1,193 | 0.541 | 1 | -0.011 |
| Y2 | mapped_resid_dtx | 16,321 | 1,193 | 0.510 | 1 | -0.042 |
| Y2 | miss_mapped_share | 16,321 | 1,193 | 0.586 | 1 | 0.034 |


## 6. Dark 470 vs invoiced 744

Train last-month companies: ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Mean miss_cp_share invoiced 0.775 vs dark 0.999 (Δ 0.223). Dark is always-missing (level). Invoiced still miss most bank CPs — not a 470-only X.

| group | n_cm | n_co | miss_mean | miss_p50 | mapped_miss | d_tx_cp |
| --- | --- | --- | --- | --- | --- | --- |
| ever_erp_744 | 13554 | 744 | 0.775 | 0.878 | 0.772 | 0.195 |
| never_erp_470 | 7603 | 470 | 0.999 | 1.000 | 0.999 | 0.002 |


## 7. Invoice CP fill vs tx CP fill (same month)

Same-month invoice CP fill mean 0.999 vs tx named share 0.252 (ρ -0.014; n=11,015). Hole is bank-book tagging, not the invoice book (Family D HHI lives on invoices). Do not invent a COMP_* ↔ COUNTERPARTY_* map.

| slice | n_cm | tx_named_p50 | tx_named_mean | inv_named_p50 | inv_named_mean |
| --- | --- | --- | --- | --- | --- |
| tx months (all train) | 20268 | 0.000 | 0.145 | — | — |
| invoice months | 11176 | 0.177 | 0.252 | 1.000 | 0.999 |
| same-month both | 11015 | 0.177 | 0.252 | 1.000 | 0.999 |
| invoiced 744 (any month) | 13062 | 0.122 | 0.225 | 1.000 | 0.999 |


## 8. Y5 fold-3 hole

Y5 AR hole (d_tx_cp_share≤0.01 ∩ hi d_n_cust) pos in fold 3: 85.9% of hole positives. High-miss AR months in fold 3: 42.3%. Without fold 3: hole 7.8% vs named 7.0% (survives=False). CLOSE as one-group — same fold-3 that owned the d_tx_cp_share Q5 sentence.

| fold | n_hole | hole_pos | P(Y5=1) hole | P(Y5=1) named | miss_hole | miss_named |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 23 | 2 | 8.7% | 5.1% | 0.997 | 0.669 |
| 1 | 6 | 2 | 33.3% | 3.2% | 1.000 | 0.496 |
| 2 | 74 | 3 | 4.1% | 9.3% | 0.999 | 0.628 |
| 3 | 253 | 55 | 21.7% | 4.4% | 0.998 | 0.633 |
| 4 | 12 | 2 | 16.7% | 7.4% | 0.997 | 0.570 |


## 9. Drop 12 chronic dark Y2 names

12 chronic dark Y2 names (GROUP_0158/0172, ≥50% b_below_0) n=12. Y2 miss_cp_share all 0.585 vs drop-12 0.572. Drop-12 does not move Y2 by ≥0.02 — not the chronic pile.

| slice | n_cm | n_co | Y2_n | Y2_pos | Y2_CV | Y3_CV | miss_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 21157 | 1214 | 16,764 | 1,236 | 0.585 | 0.554 | 0.855 |
| drop_12 | 20905 | 1202 | 16,549 | 1,060 | 0.572 | 0.559 | 0.853 |
| chronic_12 | 252 | 12 | 215 | 176 | 0.500 | LOW_POWER | 1.000 |


## 10. ICC / company-demean

miss_cp_share acf1=0.535 acf3=0.198 acf6=0.119; ICC=0.967 (sticky fill habit (BETWEEN)). Y3 company-mean 0.553 vs demean 0.482; Y2 mean 0.594 vs shock 0.516. Trait, not a month shock — not Q5 change.

| y | feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- |
| Y3 | co_mean | 5,648 | 402 | 0.553 | 1 |
| Y3 | demean | 5,536 | 372 | 0.482 | 1 |
| Y3 | now | 5,536 | 372 | 0.554 | 1 |
| Y2 | co_mean | 17,356 | 1,271 | 0.594 | 1 |
| Y2 | demean | 16,764 | 1,236 | 0.516 | 1 |
| Y2 | now | 16,764 | 1,236 | 0.585 | 1 |


## 11. Q6 lag1 / lag3 on short vs long books

Y3 miss now 0.554 / lag1 0.555 / lag3 0.551. Short so-far<12 0.545 lag1 0.557; long≥18 —. Q6 CLOSE — lag does not hold contemporaneous skill (or short books die).

| y | slice | col | lag | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Y3 | all | miss_cp_share | 0 | 5,536 | 372 | 0.554 | 1 |
| Y3 | all | miss_cp_share_lag1 | 1 | 5,546 | 383 | 0.555 | 1 |
| Y3 | all | miss_cp_share_lag3 | 3 | 4,994 | 339 | 0.551 | 1 |
| Y3 | all | d_tx_cp_share | 0 | 5,643 | 402 | 0.534 | -1 |
| Y3 | all | d_tx_cp_share_lag1 | 1 | 5,645 | 402 | 0.536 | -1 |
| Y2 | all | miss_cp_share | 0 | 16,764 | 1,236 | 0.585 | 1 |
| Y2 | all | miss_cp_share_lag1 | 1 | 15,632 | 1,120 | 0.585 | 1 |
| Y2 | all | miss_cp_share_lag3 | 3 | 13,353 | 911 | 0.579 | 1 |
| Y2 | all | d_tx_cp_share | 0 | 17,271 | 1,269 | 0.590 | -1 |
| Y2 | all | d_tx_cp_share_lag1 | 1 | 16,092 | 1,149 | 0.590 | -1 |
| Y3 | short_<12 | miss_cp_share | 0 | 3,661 | 236 | 0.545 | 1 |
| Y3 | short_<12 | miss_cp_share_lag1 | 1 | 3,664 | 240 | 0.557 | 1 |
| Y3 | short_<12 | miss_cp_share_lag3 | 3 | 3,104 | 194 | 0.560 | 1 |
| Y3 | short_<12 | d_tx_cp_share | 0 | 3,723 | 252 | 0.551 | -1 |
| Y3 | short_<12 | d_tx_cp_share_lag1 | 1 | 3,723 | 252 | 0.553 | -1 |
| Y2 | short_<12 | miss_cp_share | 0 | 10,924 | 856 | 0.568 | 1 |
| Y2 | short_<12 | miss_cp_share_lag1 | 1 | 9,773 | 739 | 0.570 | 1 |
| Y2 | short_<12 | miss_cp_share_lag3 | 3 | 7,449 | 533 | 0.565 | 1 |
| Y2 | short_<12 | d_tx_cp_share | 0 | 11,177 | 883 | 0.572 | -1 |
| Y2 | short_<12 | d_tx_cp_share_lag1 | 1 | 9,984 | 763 | 0.574 | -1 |
| Y3 | long_>=18 | miss_cp_share | 0 | 207 | 14 | LOW_POWER | 0 |
| Y3 | long_>=18 | miss_cp_share_lag1 | 1 | 206 | 14 | LOW_POWER | 0 |
| Y3 | long_>=18 | miss_cp_share_lag3 | 3 | 211 | 16 | LOW_POWER | 0 |
| Y3 | long_>=18 | d_tx_cp_share | 0 | 212 | 16 | LOW_POWER | 0 |
| Y3 | long_>=18 | d_tx_cp_share_lag1 | 1 | 212 | 16 | LOW_POWER | 0 |
| Y2 | long_>=18 | miss_cp_share | 0 | 1,778 | 117 | 0.505 | 1 |
| Y2 | long_>=18 | miss_cp_share_lag1 | 1 | 1,788 | 118 | 0.505 | 1 |
| Y2 | long_>=18 | miss_cp_share_lag3 | 3 | 1,804 | 117 | 0.613 | 1 |
| Y2 | long_>=18 | d_tx_cp_share | 0 | 1,863 | 119 | 0.504 | -1 |
| Y2 | long_>=18 | d_tx_cp_share_lag1 | 1 | 1,869 | 119 | 0.506 | -1 |


## 12. Category mix of missing-CP txs

Missing-CP txs: uncategorized 25.0% / transfer 6.6% / salary 1.8% / tax 2.2% / collection 21.8% / payment 13.2%. Not just uncat+transfer — other mapped cats also miss CP.

| category | n | |amt| | n_co | share_n | share_|amt| |
| --- | --- | --- | --- | --- | --- |
| uncategorized | 544,872 | 130,480,561,653 | 1138 | 25.0% | 29.2% |
| collection | 475,800 | 82,620,886,658 | 1187 | 21.8% | 18.5% |
| payment | 286,757 | 83,723,036,593 | 1134 | 13.2% | 18.7% |
| utility | 205,493 | 18,222,373,302 | 1059 | 9.4% | 4.1% |
| fee | 157,963 | 299,837,201 | 990 | 7.3% | 0.1% |
| transfer | 142,889 | 103,290,782,695 | 869 | 6.6% | 23.1% |
| bulk_collection | 61,903 | 415,674,838 | 242 | 2.8% | 0.1% |
| tax | 48,822 | 2,062,531,981 | 1095 | 2.2% | 0.5% |
| cash_settlement | 44,121 | 8,326,983,950 | 563 | 2.0% | 1.9% |
| pos_settlement | 41,851 | 102,424,320 | 194 | 1.9% | 0.0% |
| salary | 39,998 | 3,421,424,266 | 773 | 1.8% | 0.8% |
| bulk_payment | 37,383 | 1,150,608,509 | 471 | 1.7% | 0.3% |
| social_security | 22,821 | 555,671,930 | 648 | 1.0% | 0.1% |
| debt_repayment | 20,437 | 585,891,426 | 474 | 0.9% | 0.1% |
| cash_withdrawal | 12,806 | 8,774,958,909 | 484 | 0.6% | 2.0% |


## 13. Amounts — p50 |amt| missing vs named

p50 |amt| missing-CP 508 vs named-CP 663 (ratio 0.767). Missing-CP tickets are not systematically larger.

| kind | n | p50_|amt| | p90_|amt| | mean_|amt| | pooled_|amt| |
| --- | --- | --- | --- | --- | --- |
| named_cp | 237,992 | 663 | 18914 | 70851 | 16,862,045,141 |
| missing_cp | 2,178,115 | 508 | 31869 | 205170 | 446,883,947,805 |


## Extra 14 — condition on uncat≤0.10

Y3 miss_cp_share on uncat≤0.10 months 0.600 vs size 0.622. After conditioning on low-uncat, missing-CP does not beat size.

| y | slice | feat | n | n_pos | CV | size |
| --- | --- | --- | --- | --- | --- | --- |
| Y3 | all | miss_cp_share | 5,536 | 372 | 0.554 | 0.617 |
| Y3 | all | miss_mapped_share | 5,494 | 365 | 0.566 | 0.617 |
| Y2 | all | miss_cp_share | 16,764 | 1,236 | 0.585 | 0.552 |
| Y2 | all | miss_mapped_share | 16,321 | 1,193 | 0.586 | 0.552 |
| Y3 | uncat≤0.10 | miss_cp_share | 2,853 | 167 | 0.600 | 0.622 |
| Y3 | uncat≤0.10 | miss_mapped_share | 2,853 | 167 | 0.598 | 0.622 |
| Y2 | uncat≤0.10 | miss_cp_share | 7,729 | 451 | 0.571 | 0.525 |
| Y2 | uncat≤0.10 | miss_mapped_share | 7,729 | 451 | 0.572 | 0.525 |


## Extra 15 — condition on `d_tx_cp_share`

Y3 miss on dtx≥0.20 0.396 vs size 0.667; on the Y5 hole (dtx≤0.01) 0.494. If skill lives only on the unnamed 6m head, it is the d_tx_cp_share object.

| slice | n | n_pos | Y3_miss | Y3_size | miss_mean |
| --- | --- | --- | --- | --- | --- |
| dtx≥0.20 | 1,370 | 62 | 0.396 | 0.667 | 0.531 |
| dtx≤0.01 | 3,110 | 229 | 0.494 | 0.645 | 1.000 |
| all | 5,536 | 372 | 0.554 | 0.617 | 0.855 |


## Extra 16 — Y5 AR replica (do not change 0.611)

Y5 AR d_tx_cp_share train replica 0.611 (CONFIRM night 0.611). CV 0.576 (night y5_why CV was 0.576). Monthly miss_cp_share Y5 CV 0.586. Do not change the night 0.611 quote.

| feature | CV | train | sign | n | n_pos | folds |
| --- | --- | --- | --- | --- | --- | --- |
| miss_cp_share | 0.586 | 0.609 | 1 | 3,283 | 236 | 0.515 0.597 0.518 0.660 0.638 |
| d_tx_cp_share (replica) | 0.576 | 0.611 | -1 | 3,308 | 236 | 0.541 0.575 0.485 0.699 0.582 |
| named_share (1−miss, month) | 0.586 | 0.609 | -1 | 3,283 | 236 | 0.515 0.597 0.518 0.660 0.638 |


## Extra 17 — Y3 fold spread

Y3 miss fold AUCs 0.517 0.530 0.528 0.585 0.608 range 0.091. Fold spread <0.15 — not a one-group Y3 dummy.

| fold | auroc | n_va | n_pos | hi_miss_cm | hi_miss_pos |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.517 | 1292 | 51 | 915 | 36 |
| 1 | 0.530 | 674 | 87 | 465 | 64 |
| 2 | 0.528 | 1055 | 58 | 529 | 34 |
| 3 | 0.585 | 1348 | 74 | 859 | 56 |
| 4 | 0.608 | 1167 | 102 | 463 | 64 |


## Extra 18 — drop fold 3

Drop fold 3: Y3 miss 0.546 vs size 0.635; leftover after uncat+dtx 0.519.

## Extra 19 — monthly named vs 6m `d_tx_cp_share`

Monthly named_share vs 6m d_tx_cp_share ρ=0.947 max|Δ|=0.989 mean|Δ|=0.043. miss vs 1−d_tx ρ=0.947. Monthly missing-CP is the same object as the store fill rate (window noise).

## Extra 20 — holdout coverage only

Holdout coverage only (no AUROC): 72 companies / 1073 CM.

| col | n_cm | n_co | defined | mean | p50 |
| --- | --- | --- | --- | --- | --- |
| miss_cp_share | 1073 | 72 | 97.3% | 0.881 | 1.000 |
| miss_cp_amt | 1073 | 72 | 97.3% | 0.896 | 1.000 |
| miss_mapped_share | 1073 | 72 | 94.3% | 0.881 | 1.000 |
| d_tx_cp_share | 1073 | 72 | 99.5% | 0.097 | 0.000 |
| inv_cp_share | 1073 | 72 | 48.6% | 1.000 | 1.000 |


## Extra 21 — group ICC of company-median

Company-median miss_cp_share ICC across group_id 0.893 (k=235). Holding style if high.

## Extra 22 — Y3 inside size terciles

Y3 miss inside size terciles: skill in 1/3 bands that are defined. If only one band, it is a size-band dummy.

| size_tercile | p50_log_in3 | n | n_pos | Y3_miss | mean_miss |
| --- | --- | --- | --- | --- | --- |
| 1 | 8.879 | 1,072 | 150 | 0.448 | 0.844 |
| 2 | 12.506 | 1,974 | 93 | 0.594 | 0.823 |
| 3 | 14.724 | 2,376 | 118 | 0.542 | 0.862 |


## Extra 23 — invoiced-744 only

Invoiced-744 Y3 miss 0.609 mapped-miss 0.627 vs size 0.623.

## Extra 24 — quintiles (no Y7 X)

miss_cp_share quintiles vs Y2 / Y3 / Y5-AR base rates (descriptive; Y7 not scored).

| q | n_cm | p50 | Y2 | Y3 | Y5_AR |
| --- | --- | --- | --- | --- | --- |
| 1 | 4181 | 0.491 | 3.1% | 5.0% | 4.8% |
| 2 | 16087 | 1.000 | 8.4% | 7.1% | 9.2% |


## Extra 25 — invoiced-744 twin / leftover

Invoiced-744: miss↔d_tx ρ=-0.919 (still TWIN). Y3 miss 0.609 resid-dtx 0.540 vs size 0.623. Twin is not only the dark 1.0 pile.

| item | value | note |
| --- | --- | --- |
| ρ miss↔d_tx (744) | -0.919 | TWIN |
| ρ miss↔uncat (744) | 0.127 |  |
| Y3 miss (744) | 0.609 | n=3,564 pos=248 |
| Y3 resid d_tx (744) | 0.540 |  |
| Y3 resid both (744) | 0.534 |  |
| Y3 size (744) | 0.623 |  |


## Extra 26 — all-miss pile (qcut 2 bins)

Exact-all-miss months (share≥0.999): 12,341 / 20,268 (60.9%). qcut collapsed to 2 bins because p50=1.0. All-miss dummy Y3 0.552. Partial-only (intensity) Y3 0.385.

| slice | n_cm | n_co | dark | Y2 | Y3 | Y3_CV |
| --- | --- | --- | --- | --- | --- | --- |
| all_miss>=0.999 | 12341 | 1080 | 58.0% | 8.7% | 7.9% | 0.500 |
| partial | 7927 | 637 | 0.6% | 5.3% | 5.1% | 0.385 |
| all_defined | 20268 | 1214 | 35.6% | 7.4% | 6.7% | 0.554 |


## Extra 27 — miss vs ever_erp dummy

miss↔ever_erp ρ=-0.564 (not the 470 dummy). Y3 erp-dummy 0.460 vs miss 0.554; resid after erp 0.597.

| item | value |
| --- | --- |
| ρ miss↔ever_erp | -0.564 |
| Y3 ever_erp dummy | 0.460 |
| Y3 miss | 0.554 |
| Y3 miss resid erp | 0.597 |


## Extra 28 — named-CP category mix

Named-CP txs n=237,992. Top token `collection`. If named rows are collections/payments, the hole is still bank tagging on the rest.

| category | n | share_n |
| --- | --- | --- |
| collection | 59,420 | 25.0% |
| payment | 57,851 | 24.3% |
| uncategorized | 48,651 | 20.4% |
| utility | 42,274 | 17.8% |
| fee | 9,284 | 3.9% |
| tax | 3,415 | 1.4% |
| transfer | 3,182 | 1.3% |
| bulk_collection | 2,836 | 1.2% |
| bulk_payment | 2,753 | 1.2% |
| pos_settlement | 2,625 | 1.1% |


## Extra 29 — 360 vs 110 dark

All-dark vs mixed-dark miss level. If both ≈1.0, dark is a company-book trait, not sibling-ERP.

| group | n_cm | n_co | miss_mean | miss_p50 |
| --- | --- | --- | --- | --- |
| invoiced_744 | 13554 | 744 | 0.775 | 0.878 |
| all_dark_360 | 5660 | 360 | 1.000 | 1.000 |
| mixed_dark_110 | 1943 | 110 | 0.996 | 1.000 |


## Extra 30 — months-on-book

Months 1–3 miss 93.6% vs months 13+ 79.5%. Onboarding hole.

| so_far | n_cm | miss | Y3_CV | Y3_n_pos |
| --- | --- | --- | --- | --- |
| 1-3 | 3642 | 93.6% | LOW_POWER | 44 |
| 4-6 | 3637 | 89.3% | 0.477 | 66 |
| 7-12 | 6059 | 85.6% | 0.568 | 149 |
| 13-18 | 4590 | 81.0% | 0.562 | 113 |
| 19-24 | 3229 | 77.3% | LOW_POWER | 0 |


## Extra 31 — Q6 so-far≥13

Y3 so-far≥13 miss now 0.562 n_pos=113; lag1 0.545. Q6 stays CLOSE unless lag1 holds a contemporaneous ≥0.58.

## Extra 32 — invoice-named ∩ tx-unnamed

Invoice-named (≥0.99) ∩ tx-unnamed (≤0.01): 3,768 cm, Y5-AR 11.1% vs invoice-named ∩ tx-named>0.20 5,174 cm, Y5-AR 5.2%. Hole positives in fold 3: 79.1%. This *is* the Y5 d_tx_cp_share tagging sentence — not a new KEEP-Q5.

## Extra 33 — dark vs invoiced Y2/Y3

Dark miss is ~1.0 — AUROC on dark should die (no variance). Skill if any lives on the 744.

| slice | Y2_CV | Y2_n_pos | Y3_CV | Y3_n_pos | miss_mean |
| --- | --- | --- | --- | --- | --- |
| invoiced_744 | 0.612 | 692 | 0.609 | 248 | 0.775 |
| dark_470 | 0.505 | 544 | 0.514 | 124 | 0.999 |


## Extra 34 — ops mix missing vs named

If missing-CP and named-CP have similar ops/uncat mix, the hole is tagging, not category.

| kind | ops_coll_pay | uncat | transfer |
| --- | --- | --- | --- |
| named_cp | 51.6% | 20.4% | 1.3% |
| missing_cp | 39.6% | 25.0% | 6.6% |


## Extra 35 — company-median always-missing

Company-median miss≥0.99: 755 / 1214 companies. median↔ever_erp ρ=-0.600. Always-missing is a type, not a month.

| slice | n_co | dark | ever_Y2 |
| --- | --- | --- | --- |
| median≥0.99 | 755 | 62.1% | 15.3% |
| median<0.99 | 459 | 0.2% | 12.4% |


## Extra 36 — invoiced-744 Y2 vs size

Invoiced-744 Y2 miss 0.612 vs size 0.558 (Δ 0.054); d_tx replica 0.616; resid after d_tx 0.533. Beats size on ERP Y2 — still a fill-rate twin, not a new X.

## Extra 37 — always-missing invoiced companies by fold

Always-missing invoiced companies 286 / 744. Fold 3 holds 29.7%. Always-missing ERP names are not fold-3-only.

| fold | n_always_erp | share | n_groups |
| --- | --- | --- | --- |
| 0 | 64 | 22.4% | 17 |
| 1 | 52 | 18.2% | 17 |
| 2 | 51 | 17.8% | 18 |
| 3 | 85 | 29.7% | 16 |
| 4 | 34 | 11.9% | 17 |


## Extra 38 — leftover after ever_erp + d_tx

Y3 resid after ever_erp+d_tx 0.545 vs size 0.617 (slopes ['-0.025', '-1.017']). If this dies, missing-CP is dark-level + fill-rate, nothing leftover.

## Extra 39 — mapped-miss leftover after d_tx on 744

Invoiced-744 mapped-miss Y3 0.627 resid after d_tx 0.550 vs size 0.623. Last leftover after parking uncat *and* the 6m fill.

Plot: `missing_cp_vs_uncat.png`.

## What failed / next

- d_tx_cp_share twin ρ=-0.947 — CLOSE as (b)
- leftover after uncat/dtx dies — CLOSE as twin
- Y5 fold-3 pile again — CLOSE as (e)
- invoiced-744 still d_tx twin ρ=-0.919 — not only the 470
- Missing-CP txs: uncategorized 25.0% / transfer 6.6% / salary 1.8% / tax 2.2% / collection 21.8% / payment 13.2%. Not just uncat+transfer — other mapped cats also miss CP.
- Same-month invoice CP fill mean 0.999 vs tx named share 0.252 (ρ -0.014; n=11,015). Hole is bank-book tagging, not the invoice book (Family D HHI lives on invoices). Do not invent a COMP_* ↔ COUNTERPARTY_* map.
- Exact-all-miss months (share≥0.999): 12,341 / 20,268 (60.9%). qcut collapsed to 2 bins because p50=1.0. All-miss dummy Y3 0.552. Partial-only (intensity) Y3 0.385.
- Invoice-named (≥0.99) ∩ tx-unnamed (≤0.01): 3,768 cm, Y5-AR 11.1% vs invoice-named ∩ tx-named>0.20 5,174 cm, Y5-AR 5.2%. Hole positives in fold 3: 79.1%. This *is* the Y5 d_tx_cp_share tagging sentence — not a new KEEP-Q5.
- Invoiced-744 Y2 miss 0.612 vs size 0.558 (Δ 0.054); d_tx replica 0.616; resid after d_tx 0.533. Beats size on ERP Y2 — still a fill-rate twin, not a new X.
- Y3 resid after ever_erp+d_tx 0.545 vs size 0.617 (slopes ['-0.025', '-1.017']). If this dies, missing-CP is dark-level + fill-rate, nothing leftover.
- Invoiced-744 mapped-miss Y3 0.627 resid after d_tx 0.550 vs size 0.623. Last leftover after parking uncat *and* the 6m fill.

Elapsed 8s. Cuts: prevalence, uncat 2×2, Spearman, singles, residual, dark 470/744, invoice vs tx fill, Y5 fold-3, chronic-12, ICC, Q6, category mix, amounts, low-uncat leftover, dtx slices, Y5 replica, Y3 folds, drop fold 3, monthly vs 6m, holdout, group ICC, size terciles, invoiced-only, quintiles, 744-twin, all-miss pile, erp dummy, named cats, 360/110, so-far, Q6≥13, pure bank hole, dark Y, ops mix, always-missing companies, 744 Y2, always-ERP folds, resid erp+dtx.

Did **not**: merge parquet, invent `y_missing_cp`, score Y7, edit counterparties.py / uncat_qa.py / y5_why.py / y11_dark.py, rewrite duckdb, run `build_targets`, touch `product/`, write 0–100, change night Y3 0.762/0.752 or Y5 0.611 quotes, write the parent journal.
