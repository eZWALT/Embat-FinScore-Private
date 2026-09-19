# Unused leftover of `d_tx_cp_share` on the 44

Generated `2026-09-19T05:01:48+02:00` by agent `7f2e91c4`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_d_tx`. Do not merge a Family D column. Do not put this on the 15-col Y3 card. Y7 never uses D. Y5 never E. Y3 never B. Night Y3 **0.762/0.752**. Days **0.711**. Size **0.617**. Y7 TURNOVER **0.720 / 0.712**. Night Y5 `d_tx_cp_share` **0.611** stays a number, PARK as X. Trees stay PARK.

`d_tx_cp_share` = 6-month *named* bank-CP fill (store). `miss_cp_share` = calendar-month share of txs with null/blank `counterparty_id` (in-memory; not written).

## Headline

d_tx vs miss_cp ρ=-0.947 (CONFIRM TWIN −0.947). Y3 leftover after days OLS 0.600 rank 0.537 vs days 0.711 — CLOSE / DROP from the 44. Y5 leftover after size 0.579 leftover after miss 0.454 — PARK as X. Y5 train 0.611 (KEEP the 0.611 quote). Fold-3 hole 85.9% one-group=True drop-fold3 train 0.541 collapse=True. Q6 CLOSE. DROP d_tx_cp_share from the 44. Do not invent y_d_tx. Night Y3 0.762/0.752 and Y5 0.611 quotes unchanged.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | d_tx is PARK as a health Y. Do not invent `y_d_tx`. |
| 2 | Who is improving? | Not this 6m named-fill share. |
| 3 | Who is turning? | Y3 leftover after days 0.600 — CLOSE / DROP from the 44. A fill twin is not a turning X. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | KEEP the 0.611 quote; PARK as X. Fold-3 hole 85.9% one-group=True. Q5 sentence stays a number, not a Y3 X and not a new Family D column. |
| 6 | Months earlier? | Y5 short lag1 0.431 — CLOSE. |


## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| d_tx_cp_share as Y3 X | **CLOSE / DROP from the 44** | leftover after days OLS 0.600 rank 0.537 vs days 0.711 size 0.617; twin ρ=-0.947 |
| d_tx_cp_share as Y5 X | **PARK as X** | train 0.611 leftover after size 0.579 after miss 0.454; fold-3 one-group=True |
| Night Y5 0.611 quote | **KEEP the 0.611 quote** | this-run train 0.611 CV 0.576 |
| d_tx as a health Y | **PARK** | do not invent `y_d_tx` |
| Q6 lag1/lag3 | **CLOSE** | Y5 now 0.576 lag1 0.563; short now 0.445 lag1 0.431. Y3 lag1 0.536. Q6 CLOSE — short books die / lag does not hold. |
| miss_cp twin | **YES — CLOSE leftover** | ρ=-0.947; leftover after miss Y3 0.509 Y5 0.454 |
| d_tx_cp_share on the 44 | **DROP from the 44** | Y3 CLOSE / DROP from the 44; Y5 PARK as X; KEEP-as-X gate beat≥0.02 / leftover / not SIZE / not twin |
| 15-col Y3 card | **not added** | night quote stays 0.762 / 0.752 |
| parquet merge / new D column | **not done** | parent decides; do not invent y_d_tx |


## 1. Coverage; 470 vs 744; invoice vs tx named

Train d_tx_cp_share cov 99.2% mean 0.126 p50 0.000 eq0 57.3%. Last-month companies ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Dark d_tx mean 0.002 eq0 97.8% NaN 1.3% (defined as 0, not NaN). Same-month invoice CP fill 0.999 vs tx named 0.252 (CONFIRM 0.999 vs 0.252).

| slice | n_cm | n_co | dtx_nn | cov | eq0 | nan | mean | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train all | 21,157 | 1214 | 20,983 | 99.2% | 57.3% | 0.8% | 0.126 | 0.000 |
| ever_erp_744 | 13,554 | 744 | 13,478 | 99.4% | 34.6% | 0.6% | 0.195 | 0.106 |
| never_erp_470 | 7,603 | 470 | 7,505 | 98.7% | 97.8% | 1.3% | 0.002 | 0.000 |


| slice | n_cm | inv_named_mean | tx_named_mean | inv_named_p50 | tx_named_p50 |
| --- | --- | --- | --- | --- | --- |
| same-month both | 11015 | 0.999 | 0.252 | 1.000 | 0.177 |


## 2–3. Spearman twins — reproduce ρ −0.947

d_tx vs miss_cp ρ=-0.947 (CONFIRM −0.947). vs uncat -0.141 vs d_cust_hhi -0.113 vs d_supp_hhi -0.146 vs days 0.043 vs size 0.027. Twins: ['miss_cp_share', '1-miss_cp_share', 'named_share']. SIZE=False.

| a | b | ρ | twin_|ρ|≥0.80 | SIZE_|ρ|≥0.50 |
| --- | --- | --- | --- | --- |
| d_tx_cp_share | miss_cp_share | -0.947 | YES | — |
| d_tx_cp_share | 1-miss_cp_share | 0.947 | YES | — |
| d_tx_cp_share | named_share | 0.947 | YES | — |
| d_tx_cp_share | a_uncat_share | -0.141 | no | — |
| d_tx_cp_share | d_cust_hhi | -0.113 | no | — |
| d_tx_cp_share | d_supp_hhi | -0.146 | no | — |
| d_tx_cp_share | c_n_days_with_tx | 0.043 | no | — |
| d_tx_cp_share | log1p(a_in3) | 0.027 | no | no |
| d_tx_cp_share | d_n_cust | 0.202 | no | — |
| d_tx_cp_share | ever_erp | 0.596 | no | — |


## 4. Single-feature train group-fold AUROC

Y3 d_tx CV 0.534 vs size 0.617 (CONFIRM 0.617) vs days 0.711 (CONFIRM 0.711). Y5 AR d_tx train 0.611 (CONFIRM night 0.611) CV 0.576 (CONFIRM 0.576). Y5 size 0.469. beat_size Y3=False Y5=True.

Sign from the train side of each fold. Days bar 0.711. Size `log1p(a_in3)` 0.617. Never Y7.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_tx_cp_share | 5,643 | 402 | 0.534 | 0.541 | 0.055 | -1 | 0.449 0.532 0.534 0.551 0.601 |
| y3_recover_cash_6m | miss_cp_share | 5,536 | 372 | 0.554 | 0.556 | 0.040 | 1 | 0.517 0.530 0.528 0.585 0.608 |
| y3_recover_cash_6m | named_share | 5,536 | 372 | 0.554 | 0.556 | 0.040 | -1 | 0.517 0.530 0.528 0.585 0.608 |
| y3_recover_cash_6m | a_uncat_share | 5,536 | 372 | 0.542 | 0.534 | 0.046 | 1 | 0.528 0.542 0.553 0.607 0.478 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | d_cust_hhi | 2,485 | 141 | 0.595 | 0.585 | 0.101 | 1 | 0.740 0.454 0.581 0.608 0.590 |
| y5_ar_od30_sust | d_tx_cp_share | 3,308 | 236 | 0.576 | 0.611 | 0.079 | -1 | 0.541 0.575 0.485 0.699 0.582 |
| y5_ar_od30_sust | miss_cp_share | 3,283 | 236 | 0.586 | 0.609 | 0.067 | 1 | 0.515 0.597 0.518 0.660 0.638 |
| y5_ar_od30_sust | named_share | 3,283 | 236 | 0.586 | 0.609 | 0.067 | -1 | 0.515 0.597 0.518 0.660 0.638 |
| y5_ar_od30_sust | a_uncat_share | 3,283 | 236 | 0.539 | 0.538 | 0.079 | 1 | 0.593 0.441 0.538 0.485 0.636 |
| y5_ar_od30_sust | log1p_a_in3 | 3,315 | 237 | 0.469 | 0.504 | 0.024 | 1 | 0.510 0.461 0.455 0.450 0.469 |
| y5_ar_od30_sust | c_n_days_with_tx | 3,315 | 237 | 0.533 | 0.544 | 0.080 | 1 | 0.568 0.438 0.465 0.559 0.634 |
| y5_ar_od30_sust | d_cust_hhi | 3,311 | 237 | 0.464 | 0.521 | 0.065 | -1 | 0.526 0.439 0.487 0.506 0.364 |
| y2_neg_2of3 | d_tx_cp_share | 17,271 | 1,269 | 0.590 | 0.576 | 0.034 | -1 | 0.621 0.603 0.616 0.561 0.546 |
| y2_neg_2of3 | miss_cp_share | 16,764 | 1,236 | 0.585 | 0.572 | 0.035 | 1 | 0.620 0.604 0.607 0.553 0.543 |
| y2_neg_2of3 | named_share | 16,764 | 1,236 | 0.585 | 0.572 | 0.035 | -1 | 0.620 0.604 0.607 0.553 0.543 |
| y2_neg_2of3 | a_uncat_share | 16,764 | 1,236 | 0.584 | 0.576 | 0.031 | 1 | 0.575 0.553 0.603 0.628 0.563 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.540 | 0.046 | 1 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.577 | 0.046 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | d_cust_hhi | 7,130 | 445 | 0.564 | 0.535 | 0.161 | -1 | 0.707 0.699 0.325 0.600 0.488 |


## 5–6. Honest leftover after days / size / miss_cp

Y3 leftover after days 0.600 (lives). Y5 leftover after size 0.579 (lives). Leftover after miss_cp Y3 0.509 Y5 0.454 (dies — twins). Y3 after miss+days 0.578.

Honest screen: rank leftover after days 0.537 labeled-OLS 0.534 (panel-slope OLS 0.600 is not leftover). After miss_cp Y3 0.509 Y5 0.454 dies — twins.

| y | residual | n | n_pos | CV | train | R² | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | after days | 5,643 | 402 | 0.600 | 0.613 | 0.000 | 0.512 0.636 0.607 0.604 0.643 |
| y3_recover_cash_6m | after size | 5,523 | 391 | 0.478 | 0.505 | 0.001 | 0.421 0.538 0.516 0.482 0.433 |
| y3_recover_cash_6m | after days+size | 5,523 | 391 | 0.564 | 0.576 | 0.004 | 0.485 0.558 0.543 0.605 0.630 |
| y3_recover_cash_6m | after miss_cp | 5,536 | 372 | 0.509 | 0.514 | 0.817 | 0.463 0.546 0.478 0.548 0.513 |
| y3_recover_cash_6m | after miss+days | 5,536 | 372 | 0.578 | 0.587 | 0.817 | 0.535 0.660 0.539 0.600 0.557 |
| y3_recover_cash_6m | after uncat | 5,536 | 372 | 0.524 | 0.537 | 0.020 | 0.440 0.520 0.533 0.524 0.605 |
| y3_recover_cash_6m | after d_cust_hhi | 2,485 | 141 | 0.558 | 0.545 | 0.011 | 0.587 0.644 0.423 0.581 0.558 |
| y5_ar_od30_sust | after size | 3,308 | 236 | 0.579 | 0.614 | 0.001 | 0.541 0.578 0.483 0.715 0.580 |
| y5_ar_od30_sust | after days | 3,308 | 236 | 0.573 | 0.607 | 0.000 | 0.541 0.574 0.485 0.681 0.583 |
| y5_ar_od30_sust | after miss_cp | 3,283 | 236 | 0.454 | 0.536 | 0.817 | 0.571 0.459 0.450 0.363 0.426 |
| y5_ar_od30_sust | after miss+size | 3,283 | 236 | 0.457 | 0.536 | 0.804 | 0.571 0.459 0.453 0.373 0.430 |
| y5_ar_od30_sust | after d_cust_hhi | 3,304 | 236 | 0.578 | 0.612 | 0.011 | 0.539 0.576 0.486 0.681 0.610 |
| y2_neg_2of3 | after size | 14,883 | 1,042 | 0.621 | 0.599 | 0.001 | 0.631 0.684 0.674 0.543 0.571 |
| y2_neg_2of3 | after miss_cp | 16,764 | 1,236 | 0.528 | 0.523 | 0.817 | 0.541 0.548 0.533 0.515 0.504 |


## 7. Fold 3 / one-group hole

Y5 AR hole pos in fold 3: 85.9% of hole positives (CONFIRM one-group). Without fold 3: hole 7.8% vs named 7.0% (survives=False). Y5 d_tx train all 0.611 / drop-fold3 0.541 CV all 0.576 / drop 0.546 (0.611 collapses without fold 3). Fold-3 hole-pos groups=4 top=GROUP_0079 n=21.

| fold | n_hole | hole_pos | P(Y5=1) hole | P(Y5=1) named | n_named |
| --- | --- | --- | --- | --- | --- |
| 0 | 23 | 2 | 8.7% | 5.1% | 196 |
| 1 | 6 | 2 | 33.3% | 3.2% | 156 |
| 2 | 74 | 3 | 4.1% | 9.3% | 367 |
| 3 | 253 | 55 | 21.7% | 4.4% | 298 |
| 4 | 12 | 2 | 16.7% | 7.4% | 229 |


## 8. SIZE terciles

Y3 d_tx inside T1 0.444 T2 0.574 T3 0.517. Y5 T1 — T2 0.614 T3 0.623. dies inside T1 — not a small-firm leftover.

| y | size_tercile | p50_log_in3 | n | n_pos | CV | mean_dtx |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | T1 | 8.879 | 1,151 | 170 | 0.444 | 0.133 |
| y3_recover_cash_6m | T2 | 12.506 | 1,982 | 95 | 0.574 | 0.154 |
| y3_recover_cash_6m | T3 | 14.724 | 2,390 | 126 | 0.517 | 0.117 |
| y5_ar_od30_sust | T1 | 8.879 | 670 | 48 | LOW_POWER | 0.133 |
| y5_ar_od30_sust | T2 | 12.506 | 1,360 | 96 | 0.614 | 0.154 |
| y5_ar_od30_sust | T3 | 14.724 | 1,278 | 92 | 0.623 | 0.117 |


## 9. Q6 lag1 / lag3 on short vs long books

Y5 now 0.576 lag1 0.563; short now 0.445 lag1 0.431. Y3 lag1 0.536. Q6 CLOSE — short books die / lag does not hold.

| y | slice | col | lag | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | all | d_tx_cp_share | 0 | 5,643 | 402 | 0.534 | -1 |
| y3_recover_cash_6m | all | d_tx_cp_share_lag1 | 1 | 5,645 | 402 | 0.536 | -1 |
| y3_recover_cash_6m | all | d_tx_cp_share_lag3 | 3 | 5,078 | 355 | 0.539 | -1 |
| y3_recover_cash_6m | short_<12 | d_tx_cp_share | 0 | 3,723 | 252 | 0.551 | -1 |
| y3_recover_cash_6m | short_<12 | d_tx_cp_share_lag1 | 1 | 3,723 | 252 | 0.553 | -1 |
| y3_recover_cash_6m | short_<12 | d_tx_cp_share_lag3 | 3 | 3,153 | 205 | 0.559 | -1 |
| y3_recover_cash_6m | long_>=18 | d_tx_cp_share | 0 | 212 | 16 | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | d_tx_cp_share_lag1 | 1 | 212 | 16 | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | d_tx_cp_share_lag3 | 3 | 213 | 16 | LOW_POWER | — |
| y5_ar_od30_sust | all | d_tx_cp_share | 0 | 3,308 | 236 | 0.576 | -1 |
| y5_ar_od30_sust | all | d_tx_cp_share_lag1 | 1 | 3,308 | 236 | 0.563 | -1 |
| y5_ar_od30_sust | all | d_tx_cp_share_lag3 | 3 | 3,309 | 237 | 0.545 | -1 |
| y5_ar_od30_sust | short_<12 | d_tx_cp_share | 0 | 1,152 | 74 | 0.445 | -1 |
| y5_ar_od30_sust | short_<12 | d_tx_cp_share_lag1 | 1 | 1,153 | 74 | 0.431 | -1 |
| y5_ar_od30_sust | short_<12 | d_tx_cp_share_lag3 | 3 | 1,155 | 75 | 0.429 | -1 |
| y5_ar_od30_sust | long_>=18 | d_tx_cp_share | 0 | 718 | 67 | 0.727 | -1 |
| y5_ar_od30_sust | long_>=18 | d_tx_cp_share_lag1 | 1 | 718 | 67 | 0.711 | -1 |
| y5_ar_od30_sust | long_>=18 | d_tx_cp_share_lag3 | 3 | 718 | 67 | 0.685 | -1 |


## 10. ICC / company-demean

d_tx acf1=0.930 acf3=0.737 acf6=0.256; ICC=0.981 k=1214 (sticky fill habit (BETWEEN)). Y3 mean 0.552 vs demean 0.527; Y5 mean 0.584 vs demean 0.521. Compare missing-CP ICC 0.967.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | co_mean | 5,648 | 402 | 0.552 | 0.554 | 0.065 | -1 | 0.464 0.507 0.565 0.611 0.611 |
| y3_recover_cash_6m | demean | 5,643 | 402 | 0.527 | 0.524 | 0.056 | 1 | 0.500 0.473 0.566 0.605 0.490 |
| y3_recover_cash_6m | now | 5,643 | 402 | 0.534 | 0.541 | 0.055 | -1 | 0.449 0.532 0.534 0.551 0.601 |
| y5_ar_od30_sust | co_mean | 3,315 | 237 | 0.584 | 0.619 | 0.062 | -1 | 0.596 0.556 0.506 0.676 0.588 |
| y5_ar_od30_sust | demean | 3,308 | 236 | 0.521 | 0.530 | 0.032 | -1 | 0.476 0.517 0.514 0.564 0.534 |
| y5_ar_od30_sust | now | 3,308 | 236 | 0.576 | 0.611 | 0.079 | -1 | 0.541 0.575 0.485 0.699 0.582 |
| y2_neg_2of3 | co_mean | 17,356 | 1,271 | 0.594 | 0.578 | 0.046 | -1 | 0.653 0.620 0.599 0.556 0.542 |
| y2_neg_2of3 | demean | 17,271 | 1,269 | 0.514 | 0.517 | 0.028 | -1 | 0.478 0.493 0.544 0.525 0.531 |
| y2_neg_2of3 | now | 17,271 | 1,269 | 0.590 | 0.576 | 0.034 | -1 | 0.621 0.603 0.616 0.561 0.546 |


## 11. Dark 470: defined as 0 vs NaN

Dark 470 d_tx defined 98.7% eq0 97.8% NaN 1.3%. Y5 on 744 0.576; dark Y5 LOW_POWER / dead — needs named bank CPs on an invoice book.

| slice | n_cm | n_co | dtx_nn | dtx_eq0 | dtx_nan | mean | Y3_CV | Y5_CV | Y5_n_pos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| invoiced_744 | 13554 | 744 | 13478 | 4692 | 76 | 0.195 | 0.592 | 0.576 | 236 |
| dark_470 | 7603 | 470 | 7505 | 7437 | 98 | 0.002 | 0.488 | LOW_POWER | 0 |


## Extra — holdout coverage only

Holdout coverage only (no AUROC): 72 companies / 1073 CM.

| col | n_cm | n_co | defined | mean | p50 | eq0 |
| --- | --- | --- | --- | --- | --- | --- |
| d_tx_cp_share | 1073 | 72 | 99.5% | 0.097 | 0.000 | 66.4% |
| miss_cp_share | 1073 | 72 | 97.3% | 0.881 | 1.000 | 0.0% |
| named_share | 1073 | 72 | 97.3% | 0.119 | 0.000 | 69.7% |
| inv_cp_share | 1073 | 72 | 48.6% | 1.000 | 1.000 | 0.0% |
| a_uncat_share | 1073 | 72 | 97.3% | 0.336 | 0.213 | 18.4% |


## Extra — quintiles (no Y7 X)

d_tx quintiles bins=5. Y5 Q1 12.0% Q5 4.6% head_only (night Q5 shape).

| q | n_cm | p50 | Y2 | Y3 | Y5_AR |
| --- | --- | --- | --- | --- | --- |
| 1 | 4197 | 0.000 | 7.4% | 9.4% | 12.0% |
| 2 | 4196 | 0.000 | 9.8% | 7.2% | 11.0% |
| 3 | 4197 | 0.000 | 8.9% | 7.4% | 11.8% |
| 4 | 4196 | 0.135 | 7.3% | 6.7% | 7.6% |
| 5 | 4197 | 0.456 | 3.0% | 5.0% | 4.6% |


## Extra — monthly named vs 6m `d_tx_cp_share`

Monthly named_share vs 6m d_tx ρ=0.947 max|Δ|=0.989 mean|Δ|=0.043. miss vs 1−d_tx ρ=0.947.

## Extra — rank leftover + zero dummy

Y3 rank leftover after days 0.537. Y5 rank leftover after size 0.578; zero dummy 0.463; intensity on >0 0.580. not only a zero dummy (intensity still ranks or zero dies).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | rank leftover days | 5,643 | 402 | 0.537 | 0.536 | 0.064 | 1 | 0.616 0.575 0.541 0.507 0.448 |
| y3_recover_cash_6m | OLS leftover days | 5,643 | 402 | 0.600 | 0.613 | 0.052 | -1 | 0.512 0.636 0.607 0.604 0.643 |
| y3_recover_cash_6m | zero dummy | 5,643 | 402 | 0.515 | 0.519 | 0.062 | 1 | 0.422 0.518 0.545 0.502 0.589 |
| y3_recover_cash_6m | d_tx | >0 | 2,559 | 168 | 0.623 | 0.612 | 0.129 | -1 | 0.692 0.644 0.456 0.784 0.539 |
| y5_ar_od30_sust | rank leftover size | 3,308 | 236 | 0.578 | 0.613 | 0.084 | -1 | 0.542 0.579 0.482 0.711 0.577 |
| y5_ar_od30_sust | OLS leftover size | 3,308 | 236 | 0.579 | 0.614 | 0.085 | -1 | 0.541 0.578 0.483 0.715 0.580 |
| y5_ar_od30_sust | rank leftover miss | 3,283 | 236 | 0.433 | 0.531 | 0.074 | -1 | 0.549 0.446 0.434 0.359 0.379 |
| y5_ar_od30_sust | zero dummy | 3,308 | 236 | 0.463 | 0.569 | 0.074 | 1 | 0.488 0.521 0.451 0.339 0.514 |
| y5_ar_od30_sust | d_tx | >0 | 2,723 | 164 | 0.580 | 0.573 | 0.038 | -1 | 0.560 0.571 0.548 0.646 0.575 |


## Extra — invoiced-744 leftover

Invoiced-744: miss↔d_tx ρ=-0.919. Y3 raw 0.592 leftover-days 0.561 leftover-miss 0.535. Y5 raw 0.576 leftover-size 0.579 leftover-miss 0.457.

| y | residual | n | n_pos | CV | train | R² |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | raw | 3,618 | 264 | 0.592 | 0.585 | — |
| y3_recover_cash_6m | after days | 3,618 | 264 | 0.561 | 0.551 | 0.000 |
| y3_recover_cash_6m | after miss_cp | 3,564 | 248 | 0.535 | 0.532 | 0.771 |
| y5_ar_od30_sust | raw | 3,308 | 236 | 0.576 | 0.611 | — |
| y5_ar_od30_sust | after size | 3,308 | 236 | 0.579 | 0.614 | 0.003 |
| y5_ar_od30_sust | after miss_cp | 3,283 | 236 | 0.457 | 0.543 | 0.771 |


## Extra — Y3 fold-wise leftover after days

Y3 leftover-after-days fold spread 0.131 R2=0.000. fold leftover not a dummy.

| fold | leftover_days | n_va | n_pos |
| --- | --- | --- | --- |
| 0 | 0.512 | 1308 | 54 |
| 1 | 0.636 | 696 | 93 |
| 2 | 0.607 | 1072 | 66 |
| 3 | 0.604 | 1370 | 82 |
| 4 | 0.643 | 1197 | 107 |


## Extra — group ICC of company-median

Company-median d_tx ICC across group_id 0.897 (k=235).

## Extra — honest leftover (rank / labeled-OLS / clone)

Y3 leftover-days OLS 0.600 rank 0.537 labeled-OLS 0.534 ρ(resid,days)=0.388 ρ(resid,dtx)=0.898 (resid is not a d_tx clone). Y5 leftover-size OLS 0.579 rank 0.578. Y3 leftover after ever_erp 0.582 rank 0.564 (dark is a 0-fill). Honest leftover DIES.

| y | control | OLS | rank | labeled-OLS | ρ(resid,ctrl) | ρ(resid,dtx) | R² | honest_dies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | after days | 0.600 | 0.537 | 0.534 | 0.388 | 0.898 | 0.000 | YES |
| y3_recover_cash_6m | after size | 0.478 | 0.476 | 0.477 | -0.321 | 0.899 | 0.001 | YES |
| y3_recover_cash_6m | after miss_cp | 0.509 | 0.483 | 0.510 | -0.236 | 0.432 | 0.817 | YES |
| y3_recover_cash_6m | after ever_erp | 0.582 | 0.564 | 0.579 | -0.170 | 0.634 | 0.206 | no |
| y3_recover_cash_6m | after days+erp | 0.548 | 0.514 | 0.539 | -0.141 | 0.613 | 0.206 | YES |
| y5_ar_od30_sust | after size | 0.579 | 0.578 | 0.573 | -0.321 | 0.899 | 0.001 | no |
| y5_ar_od30_sust | after miss_cp | 0.454 | 0.433 | 0.455 | -0.236 | 0.432 | 0.817 | YES |
| y5_ar_od30_sust | after ever_erp | 0.576 | 0.576 | 0.576 | -0.170 | 0.634 | 0.206 | no |


## Extra — leftover after Family J `j_pay_match` (in-memory)

In-memory j_pay_match (not merged). ρ vs d_tx=0.262 (not a twin). Defined 9,648 train CM; dark finite 0 (must be 0). Y3 leftover after J 0.602; Y5 leftover after J 0.588. Do not merge Family J.

| y | feature | n | n_pos | CV | d_tx leftover after J | R² |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | j_pay_match | 2,646 | 151 | 0.567 | 0.602 | 0.046 |
| y5_ar_od30_sust | j_pay_match | 3,017 | 200 | 0.532 | 0.588 | 0.046 |


## Extra — drop fold 3 leftover

Drop fold 3: Y5 train 0.536 CV 0.529 leftover-size 0.545; Y3 rank leftover-days 0.487. 0.611 / leftover-size die without the one group.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y5_ar_od30_sust | raw drop-f3 | 2,249 | 140 | 0.546 | 0.541 | 0.044 | -1 | 0.541 0.575 0.485 — 0.582 |
| y5_ar_od30_sust | leftover size drop-f3 | 2,249 | 140 | 0.545 | 0.539 | 0.045 | -1 | 0.541 0.578 0.483 — 0.580 |
| y5_ar_od30_sust | leftover miss drop-f3 | 2,224 | 140 | 0.524 | 0.524 | 0.064 | 1 | 0.429 0.541 0.550 — 0.574 |
| y3_recover_cash_6m | raw drop-f3 | 4,273 | 320 | 0.529 | 0.536 | 0.062 | -1 | 0.449 0.532 0.534 — 0.601 |
| y3_recover_cash_6m | leftover days drop-f3 | 4,273 | 320 | 0.599 | 0.611 | 0.061 | -1 | 0.512 0.636 0.607 — 0.643 |
| y3_recover_cash_6m | rank leftover days drop-f3 | 4,273 | 320 | 0.487 | 0.542 | 0.087 | 1 | 0.384 0.575 0.541 — 0.448 |


## Extra — months-on-book

Months 1–3 d_tx mean 0.059 vs 13–18 0.169.

| so_far | n_cm | dtx_mean | Y3_CV | Y3_n_pos | Y5_CV | Y5_n_pos |
| --- | --- | --- | --- | --- | --- | --- |
| 1-3 | 3642 | 0.059 | LOW_POWER | 47 | LOW_POWER | 0 |
| 4-6 | 3637 | 0.082 | 0.565 | 71 | LOW_POWER | 4 |
| 7-12 | 6059 | 0.120 | 0.558 | 158 | 0.533 | 88 |
| 13-18 | 4590 | 0.169 | 0.511 | 126 | 0.542 | 87 |
| 19-24 | 3229 | 0.203 | LOW_POWER | 0 | 0.724 | 57 |


## Extra — Q6 leftover after days / size

Y3 lag1 leftover after days 0.612 short 0.640. Y5 lag1 leftover after size 0.566 short 0.430.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | lag1 leftover days all | 5,645 | 402 | 0.612 | 0.625 | 0.058 | -1 | 0.510 0.643 0.631 0.629 0.648 |
| y3_recover_cash_6m | lag1 leftover days short_<12 | 3,723 | 252 | 0.640 | 0.655 | 0.038 | -1 | 0.584 0.657 0.680 0.618 0.662 |
| y5_ar_od30_sust | lag1 leftover size all | 3,308 | 236 | 0.566 | 0.606 | 0.083 | -1 | 0.554 0.553 0.480 0.703 0.540 |
| y5_ar_od30_sust | lag1 leftover size short_<12 | 1,153 | 74 | 0.430 | 0.558 | 0.092 | -1 | 0.322 0.571 0.455 0.390 0.413 |


## Extra — fold-3 hole groups (drop top name)

Fold-3 hole-pos groups=4 top=GROUP_0079 21/55. Drop only GROUP_0079: Y5 train 0.592 vs all 0.611. fold-3 cluster (several groups), not a single name.

| group_id | hole_pos | n_co | share_of_f3_hole |
| --- | --- | --- | --- |
| GROUP_0079 | 21 | 8 | 38.2% |
| GROUP_0132 | 19 | 7 | 34.5% |
| GROUP_0081 | 14 | 5 | 25.5% |
| GROUP_0018 | 1 | 1 | 1.8% |


## Extra — invoiced-744 honest leftover

Invoiced-744 honest leftover: Y3-days OLS 0.561 rank 0.553 labeled 0.557; Y5-size rank 0.578; Y3-miss rank 0.523.

| y | control | OLS | rank | labeled | honest_dies |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | days on 744 | 0.561 | 0.553 | 0.557 | no |
| y5_ar_od30_sust | size on 744 | 0.579 | 0.578 | 0.573 | no |
| y3_recover_cash_6m | miss_cp on 744 | 0.535 | 0.523 | 0.531 | YES |


## Extra — zero vs intensity

Y5 on d_tx>0 CV 0.580 train 0.573 (night intensity 0.580). Zero dummy is not the KEEP quote.

| slice | n_cm | Y3_CV | Y3_n_pos | Y5_CV | Y5_train | Y5_n_pos |
| --- | --- | --- | --- | --- | --- | --- |
| eq0 | 12129 | 0.500 | 234 | 0.500 | 0.500 | 72 |
| >0 | 8854 | 0.623 | 168 | 0.580 | 0.573 | 164 |
| all_nn | 20983 | 0.534 | 402 | 0.576 | 0.611 | 236 |


## Extra — leftover after `d_n_cust`

Leftover after d_n_cust: Y5 OLS 0.593 rank 0.585 Y3 rank 0.541 ρ=0.202 (not a thickness twin).

| y | OLS | rank | ρ(dtx,n_cust) | honest_dies |
| --- | --- | --- | --- | --- |
| y5_ar_od30_sust | 0.593 | 0.585 | 0.202 | no |
| y3_recover_cash_6m | 0.581 | 0.541 | 0.202 | YES |


## Extra — drop the 4 fold-3 hole groups

Drop 4 fold-3 hole groups ['GROUP_0079', 'GROUP_0132', 'GROUP_0081', 'GROUP_0018']: Y5 train 0.535 CV 0.529 leftover-size 0.529 / rank 0.528. Y3 CV 0.550.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y5_ar_od30_sust | raw drop-4-groups | 2,850 | 157 | 0.529 | 0.535 | 0.054 | -1 | 0.541 0.575 0.485 0.461 0.582 |
| y3_recover_cash_6m | raw drop-4-groups | 5,306 | 376 | 0.550 | 0.553 | 0.072 | -1 | 0.449 0.532 0.534 0.635 0.601 |
| y5_ar_od30_sust | leftover size drop-4 | 2,850 | 157 | 0.529 | — | — | — | 0.541 0.578 0.483 0.462 0.580 |


## Extra — Δ / lag1 honest leftover

Δ d_tx leftover Y5 rank 0.560 Y3 rank 0.547. lag1 rank leftover Y5 0.565 Y3 0.535; short Y3 lag1 rank 0.532 (OLS short was 0.640 — rank dies).

| y | stem | OLS leftover | rank leftover | labeled | honest_dies |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | delta | 0.554 | 0.547 | 0.592 | YES |
| y5_ar_od30_sust | delta | 0.559 | 0.560 | 0.555 | no |
| y3_recover_cash_6m | lag1 | 0.612 | 0.535 | 0.534 | YES |
| y5_ar_od30_sust | lag1 | 0.566 | 0.565 | 0.560 | no |
| y3_recover_cash_6m | lag1 short rank | 0.640 | 0.532 | 0.531 | YES |


## Extra — leftover after `a_n_tx`

Leftover after a_n_tx: Y3 rank 0.532 Y5 rank 0.583 ρ=0.014.

| y | OLS | rank | ρ | honest_dies |
| --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 0.610 | 0.532 | 0.014 | YES |
| y5_ar_od30_sust | 0.566 | 0.583 | 0.014 | no |


Plot: `d_tx_leftover.png`.

## What failed / next

- Y3 leftover after days OLS 0.600 rank 0.537 — CLOSE / DROP from the 44
- leftover after miss_cp dies Y3 0.509 Y5 0.454
- fold-3 one-group hole CONFIRM — CLOSE as leave-one-group law
- DROP d_tx_cp_share from the 44
- Q6 CLOSE short lag1 0.431 rank leftover 0.532
- drop 4 hole groups ['GROUP_0079', 'GROUP_0132', 'GROUP_0081', 'GROUP_0018']: Y5 train 0.535 (same collapse as drop fold 3)
- Y3 leftover after J 0.602 Y5 0.588 ρ=0.262 — do not merge J

Elapsed 12s. Cuts: coverage/470/invoice-fill, Spearman twin, singles, leftover days/size/miss, fold-3, size terciles, Q6, ICC, dark NaN-vs-0, holdout, quintiles, monthly-vs-6m, rank/zero, 744 leftover, fold leftover, group ICC, honest leftover, J leftover, drop-f3 leftover, so-far, Q6 leftover, fold-3 groups, 744 rank, zero vs intensity, leftover after d_n_cust.

Did **not**: merge parquet, invent `y_d_tx`, score Y7, edit counterparties.py / missing_cp_qa.py / y5_why.py, rewrite duckdb, run `build_targets`, touch `product/`, write 0–100, change night Y3 0.762/0.752 or Y5 0.611 quotes, write the parent journal / LIVE / canvas, put d_tx on the 15-col card.
