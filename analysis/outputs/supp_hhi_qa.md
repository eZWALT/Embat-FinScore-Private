# Unused leftover of `d_supp_hhi` on the 44

Generated `2026-09-19T04:50:32+02:00` by agent `c9e14b20`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage / mix only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_supp_hhi`. Do not merge with Y4. Do not reopen Y4/Y5 trees. Off the 15-col Y3 card. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 TURNOVER **0.720 / 0.712**. Y7 never D. Y5 never E. Y3 never B. Do not grow TURNOVER.

`d_supp_hhi` = Herfindahl of AP invoice counterparties in the trailing 6-month window (Family D). Needs identified supplier CPs. Dark 470 stay **NaN not 0**. Javier 14: concentration is **top1**, not HHI. Y5 already PARK this column as Y5 X (size AUROC 0.663).

## Headline

**DROP** as X. Y3 **DROP** leftover-after-days 0.464. Y4 **DROP** leftover after cust_hhi_lag3 0.523 body 0.523. Y5 **PARK** leftover-after-size 0.525. not SIZE (ρ=-0.356); twin of d_supp_top1 ρ=0.987. Tail vs body: Y5 protective 0.027 vs 0.086 (CONFIRM True); Y4 body 0.523 dead like 0.445. Object: top1 rewrite; Y5 tail is protective (not a Y4 crash). 44 should lose `d_supp_hhi`. Y5 leftover 65% → 65.5% after dropping tail (Δ 0.004). Q6 CLOSE. Night quotes unchanged: Y3 0.762 / 0.752; days 0.711; size 0.617; TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_supp_hhi`. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Not a supplier-HHI gradient. |
| 3 | Who is turning? | **DROP** as Y3 X — leftover after days 0.464 vs days 0.711. |
| 4 | Dip vs fall? | **DROP** as Y4 X — leftover after cust HHI lag3 0.523; body 0.523. Do not merge with Y4. |
| 5 | Why did it change? | Y5 tail is **protective** 0.027 vs 0.086 (CONFIRM True). Object: top1 rewrite; Y5 tail is protective (not a Y4 crash). |
| 6 | Months earlier? | **CLOSE** — short lag3 Y4 0.606 lag1 0.573 — Y4 customer HHI Q6 was CLOSE / LOW_POWER 21.7%. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| d_supp_hhi as Y3 X / the 15-col card | DROP | twin of `d_supp_top1` ρ=0.987 — HHI is the weaker rewrite. Y3 0.653 leftover-after-days 0.464. Stays off the 15-col card. |
| d_supp_hhi as Y4 X | DROP | same twin. Y4 0.525 leftover after cust_hhi_lag3 0.523; body CV 0.523 (customer body 0.445). Do not merge with Y4 / do not reopen trees. |
| d_supp_hhi as Y5 X | PARK | already PARK as Y5 X (size-rank AUROC 0.663, quote 0.663). Twin of top1. Tail is protective 0.027 vs 0.086. Do not revive trees. |
| d_supp_hhi on the 44-col keep list | DROP from the 44 | twin or leftover dies. Y3 leftover 0.464; Y4 leftover 0.523 body 0.523; Y5 PARK. Javier concentration is top1. |
| same monopoly tail as Y4 customer HHI? | NO — different object | Y5 tail is protective 0.027 vs 0.086 (CONFIRM True). Y4 customer tail was a crash (22.1% vs 11.5%). Body Y4 0.523 vs customer body 0.445. |
| twin of d_supp_top1 | DROP weaker (HHI) | ρ(HHI, top1)=0.987. TWIN ≥0.80 — DROP weaker `d_supp_hhi` (Javier concentration is top1). Mean CV HHI 0.572 top1 0.566 (HHI +0.006 is not a KEEP). Y4 customer HHI↔top1 ρ 0.991. |
| y_supp_hhi / merge with Y4 | PARK | do not invent y_supp_hhi. Do not merge with Y4. |
| Q6 lag1/lag3 on short books | CLOSE | short lag3 Y4 0.606 lag1 0.573 — Y4 customer HHI Q6 was CLOSE / LOW_POWER 21.7%. |
| ICC / trait vs month shock | TRAIT | ICC 0.959 η² 0.639 k=741 (TRAIT ≥0.85). Y3 demean 0.509 mean 0.680. Y5 demean 0.529 mean 0.461. |
| Y5 leftover 65% after dropping tail | document only | AP leftover 2×2 neither 222/341 = 65.1% (quote 65.1%). After dropping the protective tail (2 pos / 73 labeled months) leftover 222/339 = 65.5% (Δ 0.004). Dropping the tail does not raise leftover (tail held almost no positives). Do not invent a new Y. |
| 2-col z-avg top1+HHI | CLOSE | Y4 z-avg was CLOSE. Do not put on the card. |


## 1. Coverage; 470 dark NaN vs invoice-book; ever-n

Train `d_supp_hhi` coverage 50.1% (10,595/21,157); ever-n 741 companies. Dark 470 (want 470): HHI non-null 0 zero-filled 0 top1 nn 0. 470 stay NaN not 0: CONFIRM. ERP n_supp==0 months 698; HHI defined on those 0 (want 0 — n=0 keeps HHI NaN). acf1=0.745 size ρ=-0.356.

| split | cm | companies | HHI nn | cov | NaN | ever-n | ever-ERP / never | dark nn / zero | ERP nn / NaN | HHI==0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 21,157 | 1,214 | 10,595 | 50.1% | 10,562 | 741 | 744 / 470 | 0 / 0 | 10,595 / 2,959 | 0 |
| holdout | 1,073 | 72 | 527 | 49.1% | 546 | 40 | 40 / 32 | 0 / 0 | 527 / 55 | 0 |


| item | value |
| --- | ---: |
| train CM / companies | 21,157 / 1,214 |
| d_supp_hhi defined | 10,595 (50.1%) |
| ever-n companies | 741 |
| never-ERP companies | 470 (want 470; confirm_470=True) |
| dark HHI non-null / zero-filled | 0 / 0 |
| dark 0-fill | NO — CONFIRM |
| ERP n_supp==0 / HHI defined there | 698 / 0 |
| holdout coverage (check only) | 49.1% |
| acf1 / acf3 | 0.745 / 0.279 |
| size ρ vs log1p(a_in3) | -0.356 |

## 2. Spearman twins (|ρ|≥0.80)

ρ vs `d_supp_top1` 0.987 n=10,595 (TWIN — HHI is the weaker rewrite). vs `d_cust_hhi` 0.158 vs `d_tx_cp_share` -0.146 vs size -0.356 (not SIZE). Javier: concentration is top1. Y4 customer HHI↔top1 ρ 0.991.

| vs | n | Spearman | Pearson | twin ≥0.80 |
| --- | --- | --- | --- | --- |
| d_cust_hhi | 8,788 | 0.158 | 0.158 | no |
| d_cust_top1 | 8,788 | 0.146 | 0.129 | no |
| d_supp_top1 | 10,595 | 0.987 | 0.977 | YES |
| d_tx_cp_share | 10,571 | -0.146 | -0.123 | no |
| log1p(a_in3) | 10,068 | -0.356 | -0.348 | no |
| c_n_days_with_tx | 10,595 | -0.442 | -0.425 | no |
| d_n_supp | 10,595 | -0.726 | -0.348 | no |
| d_cust_hhi_lag3 | 6,983 | 0.140 | 0.148 | no |
| d_supp_top1_lag3 | 8,335 | 0.746 | 0.711 | no |
| a_io_ratio | 10,068 | -0.072 | 0.005 | no |


## 3. Quintiles + >0.975 tail on Y3 / Y4 / Y5 AP

Y5 AP tail >0.975 P(Y=1)=0.027 vs rest 0.086 (quote 2.7% vs 8.6%) — CONFIRM protective. Body HHI≤0.975 CV Y3 0.661 Y4 0.523 (lag3 0.588; Y4 customer body 0.445) Y5 0.535. Y4 body dead like customer 0.445.

Train labeled cuts. Not monotone unless noted. Body CV is signed group-fold on HHI≤0.975.

| y | Q | interval | n | n_pos | P(Y=1) | median HHI |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 1 | (0.0184, 0.116] | 576 | 15 | 0.026 | 0.0749 |
| y3_recover_cash_6m | 2 | (0.116, 0.223] | 575 | 30 | 0.052 | 0.1620 |
| y3_recover_cash_6m | 3 | (0.223, 0.373] | 575 | 42 | 0.073 | 0.2900 |
| y3_recover_cash_6m | 4 | (0.373, 0.619] | 575 | 56 | 0.097 | 0.4673 |
| y3_recover_cash_6m | 5 | (0.619, 1.0] | 576 | 60 | 0.104 | 0.8496 |
| y4_ds_r_double | 1 | (0.00658, 0.101] | 233 | 18 | 0.077 | 0.0658 |
| y4_ds_r_double | 2 | (0.101, 0.185] | 232 | 30 | 0.129 | 0.1312 |
| y4_ds_r_double | 3 | (0.185, 0.306] | 232 | 45 | 0.194 | 0.2427 |
| y4_ds_r_double | 4 | (0.306, 0.475] | 232 | 32 | 0.138 | 0.4025 |
| y4_ds_r_double | 5 | (0.475, 1.0] | 232 | 35 | 0.151 | 0.7243 |
| y5_ap_od30_ownp80 | 1 | (0.00563, 0.117] | 981 | 95 | 0.097 | 0.0772 |
| y5_ap_od30_ownp80 | 2 | (0.117, 0.218] | 980 | 91 | 0.093 | 0.1631 |
| y5_ap_od30_ownp80 | 3 | (0.218, 0.342] | 982 | 73 | 0.074 | 0.2751 |
| y5_ap_od30_ownp80 | 4 | (0.342, 0.532] | 978 | 82 | 0.084 | 0.4176 |
| y5_ap_od30_ownp80 | 5 | (0.532, 1.0] | 981 | 77 | 0.078 | 0.7144 |


| y | x | n tail / pos | P(Y=1) tail | P(Y=1) rest | tail AUROC | body CV | body n / pos | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_supp_hhi | 136 / 14 | 0.103 | 0.069 | 0.512 | 0.661 | 2741 / 189 | crash |
| y4_ds_r_double | d_supp_hhi | 29 / 5 | 0.172 | 0.137 | 0.504 | 0.523 | 1132 / 155 | crash |
| y4_ds_r_double | d_supp_hhi_lag3 | 29 / 3 | 0.103 | 0.142 | 0.495 | 0.588 | 915 / 130 | protective |
| y5_ap_od30_ownp80 | d_supp_hhi | 73 / 2 | 0.027 | 0.086 | 0.494 | 0.535 | 4829 / 416 | protective |


Y5 protective tail CONFIRM vs quote 2.7% / 8.6%: **True**. Y4 body dead like customer 0.445: **True**.

## 4. Single-feature train group-fold AUROC

Sign from the train side of each fold. Seed 20260918. Night Y3 size **0.617** (replica 0.617); days **0.711** (replica 0.711). Y4 customer HHI lag3 night **0.605** (replica 0.605). Do not quote holdout.

Y3 supp HHI 0.653 vs size 0.617 (night 0.617) days 0.711 (night 0.711) top1 0.641. Y4 supp HHI 0.525 vs size 0.507 cust_hhi_lag3 0.605 (night 0.605) top1 0.528. Y5 AP supp HHI 0.539 vs size 0.556 top1 0.530; size-vs-Y on defined 0.545; size-rank AUROC 0.663 (y5_why quote 0.663; SIZE_PARK). Beat-size Y3=True Y4=False Y5=False.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_supp_hhi | 2,877 | 203 | 0.653 | 0.091 | 1 | 0.623 | 0.789 0.689 0.573 0.643 0.570 |
| y3_recover_cash_6m | d_supp_top1 | 2,877 | 203 | 0.641 | 0.102 | 1 | 0.605 | 0.787 0.702 0.561 0.613 0.542 |
| y3_recover_cash_6m | d_cust_hhi | 2,485 | 141 | 0.595 | 0.101 | 1 | 0.585 | 0.740 0.454 0.581 0.608 0.590 |
| y3_recover_cash_6m | d_cust_hhi_lag3 | 1,862 | 115 | 0.547 | 0.084 | 1 | 0.533 | 0.638 0.457 0.478 0.533 0.629 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | d_n_supp | 3,003 | 221 | 0.699 | 0.116 | -1 | 0.685 | 0.820 0.722 0.514 0.764 0.674 |
| y3_recover_cash_6m | d_tx_cp_share | 5,643 | 402 | 0.534 | 0.055 | -1 | 0.541 | 0.449 0.532 0.534 0.551 0.601 |
| y3_recover_cash_6m | d_supp_hhi_lag1 | 2,695 | 195 | 0.648 | 0.083 | 1 | 0.616 | 0.767 0.697 0.593 0.615 0.566 |
| y3_recover_cash_6m | d_supp_hhi_lag3 | 2,199 | 161 | 0.635 | 0.079 | 1 | 0.608 | 0.741 0.688 0.599 0.600 0.543 |
| y4_ds_r_double | d_supp_hhi | 1,161 | 160 | 0.525 | 0.091 | 1 | 0.556 | 0.611 0.492 0.404 0.621 0.497 |
| y4_ds_r_double | d_supp_top1 | 1,161 | 160 | 0.528 | 0.095 | 1 | 0.552 | 0.613 0.519 0.385 0.617 0.507 |
| y4_ds_r_double | d_cust_hhi | 1,047 | 137 | 0.583 | 0.074 | 1 | 0.574 | 0.614 0.648 0.523 0.644 0.487 |
| y4_ds_r_double | d_cust_hhi_lag3 | 848 | 116 | 0.605 | 0.044 | 1 | 0.592 | 0.677 0.610 0.575 0.563 0.598 |
| y4_ds_r_double | log1p(a_in3) | 2,370 | 329 | 0.507 | 0.018 | -1 | 0.506 | 0.478 0.516 0.511 0.526 0.505 |
| y4_ds_r_double | c_n_days_with_tx | 2,370 | 329 | 0.553 | 0.046 | -1 | 0.562 | 0.565 0.576 0.471 0.573 0.579 |
| y4_ds_r_double | d_n_supp | 1,218 | 171 | 0.558 | 0.099 | -1 | 0.594 | 0.609 0.458 0.448 0.667 0.606 |
| y4_ds_r_double | d_tx_cp_share | 2,370 | 329 | 0.496 | 0.030 | -1 | 0.505 | 0.498 0.493 0.542 0.489 0.457 |
| y4_ds_r_double | d_supp_hhi_lag1 | 1,097 | 154 | 0.559 | 0.070 | 1 | 0.576 | 0.597 0.587 0.456 0.634 0.523 |
| y4_ds_r_double | d_supp_hhi_lag3 | 944 | 133 | 0.570 | 0.052 | 1 | 0.576 | 0.594 0.600 0.543 0.622 0.492 |
| y5_ap_od30_ownp80 | d_supp_hhi | 4,902 | 418 | 0.539 | 0.044 | -1 | 0.526 | 0.536 0.575 0.533 0.470 0.579 |
| y5_ap_od30_ownp80 | d_supp_top1 | 4,902 | 418 | 0.530 | 0.043 | -1 | 0.518 | 0.526 0.565 0.523 0.464 0.571 |
| y5_ap_od30_ownp80 | d_cust_hhi | 4,460 | 381 | 0.439 | 0.095 | 1 | 0.505 | 0.340 0.479 0.561 0.470 0.343 |
| y5_ap_od30_ownp80 | d_cust_hhi_lag3 | 4,214 | 349 | 0.433 | 0.078 | 1 | 0.502 | 0.346 0.460 0.529 0.470 0.361 |
| y5_ap_od30_ownp80 | log1p(a_in3) | 4,905 | 418 | 0.556 | 0.084 | 1 | 0.545 | 0.674 0.539 0.466 0.495 0.606 |
| y5_ap_od30_ownp80 | c_n_days_with_tx | 4,905 | 418 | 0.540 | 0.083 | 1 | 0.540 | 0.641 0.449 0.473 0.530 0.607 |
| y5_ap_od30_ownp80 | d_n_supp | 4,905 | 418 | 0.584 | 0.064 | 1 | 0.561 | 0.676 0.560 0.566 0.506 0.612 |
| y5_ap_od30_ownp80 | d_tx_cp_share | 4,894 | 418 | 0.532 | 0.095 | -1 | 0.552 | 0.581 0.384 0.489 0.604 0.600 |
| y5_ap_od30_ownp80 | d_supp_hhi_lag1 | 4,902 | 418 | 0.549 | 0.055 | -1 | 0.533 | 0.545 0.591 0.554 0.459 0.598 |
| y5_ap_od30_ownp80 | d_supp_hhi_lag3 | 4,696 | 401 | 0.487 | 0.061 | -1 | 0.517 | 0.524 0.406 0.503 0.444 0.558 |


## 5. Honest leftover after the bar

OLS residual of `d_supp_hhi` on the bar (train-defined slope). Leftover <0.55 dies. Y5 leftover is diagnostic only — do not revive trees.

Y3 leftover after days 0.464 (dies <0.55); after top1 0.525. Y4 leftover after cust_hhi_lag3 0.523 (dies <0.55); after top1 0.534. Y5 leftover after size 0.525 (dies <0.55); after tail-flag 0.536 after top1 0.530. Rank-ortho days 0.560 cust 0.539 size 0.475 top1 0.582.

| y | residual | n | n_pos | CV | R² | folds |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | after days | 2,877 | 203 | 0.464 | 0.181 | 0.294 0.561 0.514 0.503 0.448 |
| y3_recover_cash_6m | after size | 2,848 | 198 | 0.616 | 0.121 | 0.724 0.688 0.535 0.611 0.521 |
| y3_recover_cash_6m | after days+size | 2,848 | 198 | 0.472 | 0.206 | 0.311 0.576 0.501 0.519 0.451 |
| y3_recover_cash_6m | after d_supp_top1 | 2,877 | 203 | 0.525 | 0.955 | 0.540 0.366 0.454 0.655 0.608 |
| y3_recover_cash_6m | after d_cust_hhi | 2,464 | 138 | 0.619 | 0.025 | 0.806 0.672 0.555 0.552 0.508 |
| y3_recover_cash_6m | rank-resid after days | 2,877 | 203 | 0.560 | — | 0.700 0.588 0.535 0.524 0.454 |
| y4_ds_r_double | after d_cust_hhi_lag3 | 828 | 111 | 0.523 | 0.022 | 0.586 0.565 0.359 0.610 0.493 |
| y4_ds_r_double | after d_cust_hhi | 1,029 | 133 | 0.471 | 0.025 | 0.578 0.382 0.319 0.557 0.518 |
| y4_ds_r_double | after d_supp_top1 | 1,161 | 160 | 0.534 | 0.955 | 0.527 0.676 0.382 0.572 0.514 |
| y4_ds_r_double | after size | 1,161 | 160 | 0.531 | 0.121 | 0.633 0.500 0.393 0.620 0.509 |
| y4_ds_r_double | after tail-flag | 1,161 | 160 | 0.522 | 0.314 | 0.602 0.474 0.407 0.623 0.503 |
| y4_ds_r_double | rank-resid after cust_hhi_lag3 | 828 | 111 | 0.539 | — | 0.601 0.588 0.384 0.620 0.501 |
| y5_ap_od30_ownp80 | after size | 4,902 | 418 | 0.525 | 0.121 | 0.498 0.555 0.547 0.472 0.555 |
| y5_ap_od30_ownp80 | after tail-flag | 4,902 | 418 | 0.536 | 0.314 | 0.532 0.572 0.539 0.466 0.570 |
| y5_ap_od30_ownp80 | after size+tail | 4,902 | 418 | 0.529 | 0.343 | 0.509 0.561 0.556 0.467 0.551 |
| y5_ap_od30_ownp80 | after d_supp_top1 | 4,902 | 418 | 0.530 | 0.955 | 0.567 0.481 0.507 0.572 0.520 |
| y5_ap_od30_ownp80 | after days | 4,902 | 418 | 0.449 | 0.181 | 0.471 0.407 0.451 0.452 0.463 |
| y5_ap_od30_ownp80 | rank-resid after size | 4,902 | 418 | 0.475 | — | 0.476 0.429 0.548 0.468 0.455 |


## 6. Twin vs `d_supp_top1`

ρ(HHI, top1)=0.987. TWIN ≥0.80 — DROP weaker `d_supp_hhi` (Javier concentration is top1). Mean CV HHI 0.572 top1 0.566 (HHI +0.006 is not a KEEP). Y4 customer HHI↔top1 ρ 0.991.

## 7. SIZE terciles — tail inside T1?

Y5 T1 tail P(Y=1)=0.038 vs rest 0.070 (n_tail=53 pos=2). Protective tail survives inside T1. Y3 T1 raw 0.423 Y4 T1 raw —.

| y | tercile | n | n_pos | n tail / pos | P(Y=1) tail | P(Y=1) rest | raw CV | body CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | T1 | 585 | 83 | 64 / 8 | 0.125 | 0.144 | 0.423 | 0.414 |
| y3_recover_cash_6m | T2 | 1063 | 51 | 34 / 0 | 0.000 | 0.050 | 0.657 | 0.683 |
| y3_recover_cash_6m | T3 | 1200 | 64 | 34 / 6 | 0.176 | 0.050 | 0.664 | 0.660 |
| y4_ds_r_double | T1 | 183 | 19 | 10 / 1 | 0.100 | 0.104 | LOW_POWER | LOW_POWER |
| y4_ds_r_double | T2 | 437 | 73 | 11 / 3 | 0.273 | 0.164 | 0.446 | 0.431 |
| y4_ds_r_double | T3 | 541 | 68 | 8 / 1 | 0.125 | 0.126 | 0.547 | 0.554 |
| y5_ap_od30_ownp80 | T1 | 1315 | 90 | 53 / 2 | 0.038 | 0.070 | 0.610 | 0.607 |
| y5_ap_od30_ownp80 | T2 | 1868 | 153 | 15 / 0 | 0.000 | 0.083 | 0.431 | 0.434 |
| y5_ap_od30_ownp80 | T3 | 1719 | 175 | 5 / 0 | 0.000 | 0.102 | 0.465 | 0.466 |


## 8. Q6 lag1 / lag3 on short books

Y4 short lag3 0.606 lag1 0.573 (Y4 Q6 customer HHI was CLOSE / LOW_POWER 21.7%). Y4 all lag3 0.570. Y5 short lag1 0.526. Q6 stays CLOSE unless a short-book lag clears size+0.02 — it does not.

| y | slice | feature | n | n_pos | CV | sd | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | short_<12 | d_supp_hhi | 1,703 | 113 | 0.631 | 0.098 | 1 |
| y3_recover_cash_6m | short_<12 | d_supp_hhi_lag1 | 1,516 | 102 | 0.650 | 0.091 | 1 |
| y3_recover_cash_6m | short_<12 | d_supp_hhi_lag3 | 1,005 | 68 | 0.676 | 0.095 | 1 |
| y3_recover_cash_6m | long_>=18 | d_supp_hhi | 122 | 10 | LOW_POWER | — | — |
| y3_recover_cash_6m | long_>=18 | d_supp_hhi_lag1 | 122 | 10 | LOW_POWER | — | — |
| y3_recover_cash_6m | long_>=18 | d_supp_hhi_lag3 | 121 | 9 | LOW_POWER | — | — |
| y3_recover_cash_6m | all | d_supp_hhi | 2,877 | 203 | 0.653 | 0.091 | 1 |
| y3_recover_cash_6m | all | d_supp_hhi_lag1 | 2,695 | 195 | 0.648 | 0.083 | 1 |
| y3_recover_cash_6m | all | d_supp_hhi_lag3 | 2,199 | 161 | 0.635 | 0.079 | 1 |
| y4_ds_r_double | short_<12 | d_supp_hhi | 552 | 78 | 0.472 | 0.119 | 1 |
| y4_ds_r_double | short_<12 | d_supp_hhi_lag1 | 491 | 72 | 0.573 | 0.124 | 1 |
| y4_ds_r_double | short_<12 | d_supp_hhi_lag3 | 343 | 54 | 0.606 | 0.139 | 1 |
| y4_ds_r_double | long_>=18 | d_supp_hhi | 167 | 18 | LOW_POWER | — | — |
| y4_ds_r_double | long_>=18 | d_supp_hhi_lag1 | 167 | 18 | LOW_POWER | — | — |
| y4_ds_r_double | long_>=18 | d_supp_hhi_lag3 | 165 | 18 | LOW_POWER | — | — |
| y4_ds_r_double | all | d_supp_hhi | 1,161 | 160 | 0.525 | 0.091 | 1 |
| y4_ds_r_double | all | d_supp_hhi_lag1 | 1,097 | 154 | 0.559 | 0.070 | 1 |
| y4_ds_r_double | all | d_supp_hhi_lag3 | 944 | 133 | 0.570 | 0.052 | 1 |
| y5_ap_od30_ownp80 | short_<12 | d_supp_hhi | 1,776 | 142 | 0.519 | 0.054 | -1 |
| y5_ap_od30_ownp80 | short_<12 | d_supp_hhi_lag1 | 1,776 | 142 | 0.526 | 0.059 | -1 |
| y5_ap_od30_ownp80 | short_<12 | d_supp_hhi_lag3 | 1,572 | 125 | 0.500 | 0.045 | -1 |
| y5_ap_od30_ownp80 | long_>=18 | d_supp_hhi | 995 | 114 | 0.601 | 0.103 | -1 |
| y5_ap_od30_ownp80 | long_>=18 | d_supp_hhi_lag1 | 996 | 114 | 0.601 | 0.101 | -1 |
| y5_ap_od30_ownp80 | long_>=18 | d_supp_hhi_lag3 | 995 | 114 | 0.493 | 0.106 | -1 |
| y5_ap_od30_ownp80 | all | d_supp_hhi | 4,902 | 418 | 0.539 | 0.044 | -1 |
| y5_ap_od30_ownp80 | all | d_supp_hhi_lag1 | 4,902 | 418 | 0.549 | 0.055 | -1 |
| y5_ap_od30_ownp80 | all | d_supp_hhi_lag3 | 4,696 | 401 | 0.487 | 0.061 | -1 |


## 9. ICC / company-demean (trait vs month shock)

ICC 0.959 η² 0.639 k=741 (TRAIT ≥0.85). Y3 demean 0.509 mean 0.680. Y5 demean 0.529 mean 0.461.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | company-demean | 2,877 | 203 | 0.509 | 0.100 | -1 | 0.534 | 0.331 0.555 0.567 0.550 0.543 |
| y3_recover_cash_6m | company-mean | 3,618 | 264 | 0.680 | 0.062 | 1 | 0.668 | 0.731 0.757 0.642 0.666 0.607 |
| y4_ds_r_double | company-demean | 1,161 | 160 | 0.403 | 0.084 | -1 | 0.503 | 0.472 0.274 0.403 0.482 0.385 |
| y4_ds_r_double | company-mean | 1,405 | 195 | 0.527 | 0.119 | 1 | 0.542 | 0.519 0.685 0.384 0.597 0.451 |
| y5_ap_od30_ownp80 | company-demean | 4,902 | 418 | 0.529 | 0.050 | -1 | 0.529 | 0.502 0.500 0.586 0.578 0.477 |
| y5_ap_od30_ownp80 | company-mean | 4,905 | 418 | 0.461 | 0.049 | -1 | 0.510 | 0.540 0.432 0.473 0.441 0.418 |


## 10. Dark 470 — HHI defined?

Dark 470 HHI defined 0 zero 0. CONFIRM NaN not 0 — HHI needs invoice CPs.. COMP_0962 HHI nn=0 (refund-only ghost; live-dark).

| slice | cm | HHI nn | HHI==0 | top1 nn | cust HHI nn |
| --- | --- | --- | --- | --- | --- |
| dark 470 CM | 7,603 | 0 | 0 | 0 | 0 |
| invoice-book CM | 13,554 | 10595 | 0 | 10595 | 8928 |
| COMP_0962 (refund ghost) | 13 | 0 | 0 | 0 | 0 |


## 11. Y5 leftover 65% after dropping the protective tail

AP leftover 2×2 neither 222/341 = 65.1% (quote 65.1%). After dropping the protective tail (2 pos / 73 labeled months) leftover 222/339 = 65.5% (Δ 0.004). Dropping the tail does not raise leftover (tail held almost no positives). Do not invent a new Y.

| slice | n 2×2 | cash_only | hhi_only | both | neither | leftover |
| --- | --- | --- | --- | --- | --- | --- |
| all AP pos (quote 65.1%) | 341 | 70 | 37 | 12 | 222 | 65.1% |
| drop HHI>0.975 pos | 339 | 70 | 35 | 12 | 222 | 65.5% |


Do not invent a new Y from the leftover cell.

## Extra A. Holdout coverage / mix (LOW_POWER)

Holdout 72 is coverage / mix only (Y4 train crash 80% / holdout flipped 38% crash / 88% spike — LOW_POWER). Do not quote holdout AUROC. y3_recover_cash_6m labeled 235 pos 14 HHI-nn 92. y4_ds_r_double labeled 135 pos 16 HHI-nn 35. y5_ap_od30_ownp80 labeled 195 pos 14 HHI-nn 195.

| y | n labeled | n_pos | HHI nn | n tail / pos | P(Y=1) tail | P(Y=1) rest |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 235 | 14 | 92 | 0 / 0 | — | 0.054 |
| y4_ds_r_double | 135 | 16 | 35 | 0 / 0 | — | 0.057 |
| y5_ap_od30_ownp80 | 195 | 14 | 195 | 0 / 0 | — | 0.072 |


## Extra B. 2-col z-avg of top1 + HHI

2-col z-avg of `d_supp_hhi` + `d_supp_top1` is a twin stack. Y4 customer z-avg was CLOSE. Do not put this on any card. Y3 0.647 Y4 0.526 Y5 0.534.

| y | zavg CV | HHI-CC | top1-CC | gap vs HHI | just HHI | n / pos |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 0.647 | 0.653 | 0.641 | -0.006 | True | 2877 / 203 |
| y4_ds_r_double | 0.526 | 0.525 | 0.528 | 0.001 | True | 1161 / 160 |
| y5_ap_od30_ownp80 | 0.534 | 0.539 | 0.530 | -0.005 | True | 4902 / 418 |


CLOSE. Do not put on the card.

## Extra C. Same-n leftover after top1

Same-n HHI vs top1: Y3 leftover after top1 0.525 R²=0.955. Near-identity — leftover after top1 is not a new object.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | HHI same-n | 2,877 | 203 | 0.653 | 0.091 | 1 | 0.623 | 0.789 0.689 0.573 0.643 0.570 |
| y3_recover_cash_6m | top1 same-n | 2,877 | 203 | 0.641 | 0.102 | 1 | 0.605 | 0.787 0.702 0.561 0.613 0.542 |
| y3_recover_cash_6m | HHI resid after top1 | 2,877 | 203 | 0.525 | 0.117 | 1 | 0.548 | 0.540 0.366 0.454 0.655 0.608 |
| y4_ds_r_double | HHI same-n | 1,161 | 160 | 0.525 | 0.091 | 1 | 0.556 | 0.611 0.492 0.404 0.621 0.497 |
| y4_ds_r_double | top1 same-n | 1,161 | 160 | 0.528 | 0.095 | 1 | 0.552 | 0.613 0.519 0.385 0.617 0.507 |
| y4_ds_r_double | HHI resid after top1 | 1,161 | 160 | 0.534 | 0.106 | -1 | 0.529 | 0.527 0.676 0.382 0.572 0.514 |
| y5_ap_od30_ownp80 | HHI same-n | 4,902 | 418 | 0.539 | 0.044 | -1 | 0.526 | 0.536 0.575 0.533 0.470 0.579 |
| y5_ap_od30_ownp80 | top1 same-n | 4,902 | 418 | 0.530 | 0.043 | -1 | 0.518 | 0.526 0.565 0.523 0.464 0.571 |
| y5_ap_od30_ownp80 | HHI resid after top1 | 4,902 | 418 | 0.530 | 0.039 | -1 | 0.526 | 0.567 0.481 0.507 0.572 0.520 |


## Extra D. Tail companies on non-tail months

Companies that ever hit supp HHI>0.975: 208. If those same books are quiet on non-tail months, the object is the bin, not a gradient.

| y | tail cos | tail months / pos | P(Y=1) tail mo | non-tail mo / pos | P(Y=1) non-tail mo |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 113 | 136 / 14 | 0.103 | 526 / 73 | 0.139 |
| y4_ds_r_double | 33 | 29 / 5 | 0.172 | 161 / 30 | 0.186 |
| y5_ap_od30_ownp80 | 119 | 73 / 2 | 0.027 | 597 / 62 | 0.104 |


## Extra E. Size-rank AUROC (y5_why 0.663 replica)

Y5 AP size-rank AUROC of `d_supp_hhi` 0.663 (y5_why quote 0.663) — CONFIRM. This is large-vs-small ranking, not size-vs-Y. PARK as Y5 X stands.

| y | feature | size-rank AUROC | size ρ | n | SIZE_PARK |
| --- | --- | --- | --- | --- | --- |
| y5_ap_od30_ownp80 | d_supp_hhi | 0.663 | -0.331 | 4,902 | True |
| y5_ap_od30_ownp80 | d_supp_top1 | 0.642 | -0.291 | 4,902 | True |
| y5_ap_od30_ownp80 | d_n_supp | 0.799 | 0.577 | 4,905 | True |
| y3_recover_cash_6m | d_supp_hhi | 0.622 | -0.263 | 2,848 | True |
| y4_ds_r_double | d_supp_hhi | 0.608 | -0.239 | 1,161 | True |


## Extra F. Y3 T2/T3 pocket leftover after days

Y3 T2+T3 HHI 0.666 leftover-after-days 0.417 vs days 0.688. Pocket dies after days — the 0.653 is activity, not leftover turning.

| slice | n / pos | HHI CV | days CV | leftover days |
| --- | --- | --- | --- | --- |
| T1 | 585 / 83 | 0.423 | 0.615 | 0.446 |
| T2 | 1063 / 51 | 0.657 | 0.634 | 0.360 |
| T3 | 1200 / 64 | 0.664 | 0.726 | 0.401 |
| T2+T3 | 2263 / 115 | 0.666 | 0.688 | 0.417 |


## Extra G. Leftover after `d_n_supp`

ρ(HHI, n_supp)=-0.726. Y3 n_supp 0.699 leftover after n_supp 0.637 R²=0.121 — not a linear count rewrite. Extra L: leftover after n_supp+days dies (days was leaking through the 0.637).

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_n_supp | 3,003 | 221 | 0.699 | 0.116 | -1 | 0.685 | 0.820 0.722 0.514 0.764 0.674 |
| y3_recover_cash_6m | HHI resid after n_supp | 2,877 | 203 | 0.637 | 0.085 | 1 | 0.602 | 0.767 0.658 0.627 0.596 0.537 |
| y4_ds_r_double | d_n_supp | 1,218 | 171 | 0.558 | 0.099 | -1 | 0.594 | 0.609 0.458 0.448 0.667 0.606 |
| y4_ds_r_double | HHI resid after n_supp | 1,161 | 160 | 0.458 | 0.087 | 1 | 0.519 | 0.597 0.448 0.365 0.413 0.469 |
| y5_ap_od30_ownp80 | d_n_supp | 4,905 | 418 | 0.584 | 0.064 | 1 | 0.561 | 0.676 0.560 0.566 0.506 0.612 |
| y5_ap_od30_ownp80 | HHI resid after n_supp | 4,902 | 418 | 0.460 | 0.025 | 1 | 0.500 | 0.486 0.435 0.484 0.435 0.460 |


## Extra H. Company-mean leftover after days + inverse days

Y3 company-mean HHI 0.680 leftover after days 0.580 R²=0.230. Inverse: days after HHI 0.722 R²=0.181 (days 0.711 should survive). Style leftover still ranks.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | company-mean HHI | 3,618 | 264 | 0.680 | 0.062 | 1 | 0.668 | 0.731 0.757 0.642 0.666 0.607 |
| y3_recover_cash_6m | mean resid after days | 3,618 | 264 | 0.580 | 0.084 | 1 | 0.561 | 0.656 0.653 0.603 0.523 0.464 |
| y3_recover_cash_6m | days resid after HHI | 2,877 | 203 | 0.722 | 0.088 | -1 | 0.733 | 0.664 0.784 0.599 0.811 0.750 |


## Extra I. Holdout tail mix (LOW_POWER)

Holdout Y4 positives 16 (train mix flipped 38% crash / 88% spike — LOW_POWER). Do not quote holdout AUROC. Tail n is a coverage check only.

| y | hold n / pos | HHI nn | tail n / pos | P(Y=1) tail | P(Y=1) rest |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 235 / 14 | 92 | 0 / 0 | — | 0.054 |
| y4_ds_r_double | 135 / 16 | 35 | 0 / 0 | — | 0.057 |
| y5_ap_od30_ownp80 | 195 / 14 | 195 | 0 / 0 | — | 0.072 |


## Extra J. Y3 body leftover after days

Y3 body HHI 0.661 leftover-after-days 0.411 vs days 0.741 R²=0.181. Body gradient dies after days — not a monopoly-tail leftover.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | HHI body ≤0.975 | 2,741 | 189 | 0.661 | 0.107 | 1 | 0.624 | 0.832 0.696 0.597 0.607 0.570 |
| y3_recover_cash_6m | days on body | 2,741 | 189 | 0.741 | 0.077 | -1 | 0.744 | 0.725 0.821 0.618 0.783 0.760 |
| y3_recover_cash_6m | HHI body resid after days | 2,741 | 189 | 0.411 | 0.082 | -1 | 0.502 | 0.266 0.438 0.466 0.450 0.434 |


## Extra K. Short-book lag3 leftover (Q6 honesty)

Y4 short lag3 0.606 leftover after cust_hhi_lag3 —. Y3 short lag3 leftover after days 0.590. Q6 CLOSE — short-book lag is not leftover after the honest bar.

| y | bar | lag3 CV | leftover | n / pos | R² |
| --- | --- | --- | --- | --- | --- |
| y4_ds_r_double | cust_hhi_lag3 | 0.606 | LOW_POWER | 343 / 54 | 0.026 |
| y3_recover_cash_6m | days | 0.676 | 0.590 | 1005 / 68 | 0.145 |
| y5_ap_od30_ownp80 | size | 0.500 | 0.481 | 1572 / 125 | 0.106 |


## Extra L. Leftover after n_supp + days

Y3 leftover after n_supp+days 0.470 R²=0.230 (after n_supp-only was 0.637). Rank-ortho 0.401. Days-leftover folds 0.294 0.561 0.514 0.503 0.448. Dies after n_supp+days — count+activity eat the 0.653.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | HHI resid after n_supp+days | 2,877 | 203 | 0.470 | 0.107 | 1 | 0.512 | 0.298 0.564 0.549 0.495 0.446 |
| y3_recover_cash_6m | rank resid n_supp then days | 2,877 | 203 | 0.401 | 0.044 | -1 | 0.515 | 0.350 0.370 0.464 0.408 0.415 |
| y3_recover_cash_6m | OLS resid after days (folds) | 2,877 | 203 | 0.464 | 0.103 | 1 | 0.507 | 0.294 0.561 0.514 0.503 0.448 |


## What failed / next (held for wave note)

- twin of d_supp_top1 ρ=0.987 — DROP weaker HHI
- Y3 leftover after days 0.464 dies <0.55
- Y4 leftover after cust_hhi_lag3 0.523; body 0.523
- Y5 size-rank AUROC 0.663 ≥0.60 — PARK as Y5 X
- Y5 tail CONFIRM protective — not a Y4-style crash
- dropping the protective tail does not raise the 65% leftover (almost no tail pos)
- Y3 T2+T3 leftover after days 0.417 dies — 0.653 is activity
- Y3 body leftover after days 0.411 dies
- Y5 size-rank AUROC 0.663 CONFIRM vs 0.663
- Q6 short lag3 leftover after the honest bar dies — CLOSE
- Y3 leftover after n_supp+days 0.470 dies

Elapsed 7s. Cuts 1–12 plus extras (holdout mix, z-avg, same-n leftover, tail-company months, size-rank, Y3 T2/T3 pocket, n_supp leftover, style, holdout tail, Y3 body leftover, short-book Q6 leftover, n_supp+days).

## What this module did not do

- Did not change night Y3 0.762 / 0.752, days 0.711, size 0.617, or Y7 TURNOVER 0.720 / 0.712.
- Did not put supp HHI on the 15-col Y3 card. Did not grow TURNOVER.
- Did not reopen Y4/Y5 trees. Did not invent `y_supp_hhi`. Did not merge with Y4.
- Did not use Family E as Y5 X. Did not use Family B as Y3 X. Did not use Family F as Y4 X.
- Did not write 0–100 / pillars. Did not touch `product/`.
- Did not rewrite parquet or duckdb. Did not merge Family I/M/J. Did not run `build_targets`.
- Did not fit on holdout 72. Did not commit. Did not write the parent journal / LIVE / canvas.
