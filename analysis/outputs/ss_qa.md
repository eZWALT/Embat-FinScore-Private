# Unused leftover of `c_ss_month` after `c_n_days_with_tx`

Generated `2026-09-19T05:36:41+02:00` by agent `d4c8e201`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_ss`. Do not edit `ops.py` / the 15-col card. Y3 never B. Do not overwrite `n_tx_qa.*` / `salary_qa.*` / `tax_qa.*`. Night Y3 **0.762 / 0.752**. Days **0.711**. Size **0.617**. Salary **0.671**. Y7 TURNOVER **0.72 / 0.712**.

`c_ss_month` = 1 if any category = social_security this month (fillna 0). Perm-stable #1 (ΔAUROC 0.034, sign −). `c_missed_salary` CLOSE as Y3 X (0.513). `a_n_tx` DROPPED as a days twin.

## Headline

KEEP leftover-after-days rank 0.635 (OLS 0.635, fake_ols=False). Y3 SS 0.693 vs days 0.711 vs size 0.617 vs salary 0.671. After salary 0.668. Inverse 0.641. 15-col card: KEEP. Q6 KEEP. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | SS is a payroll-presence flag, not a health Y. Do not invent y_ss. |
| 2 | Who is improving? | Y3 leftover after days 0.635 — KEEP. Quiet-stressed recover is the SHAP story; leftover asks if SS is more than days. |
| 3 | Who is turning? | Y2 SS 0.539. Missed-SS is in-memory only. |
| 4 | Dip vs fall? | Not this binary month flag. |
| 5 | Why did it change? | Perm #1 ΔAUROC 0.034. Leftover after days 0.635; after salary 0.668. Calendar monthly. |
| 6 | Months earlier? | lag1 leftover after days_lag1 0.631 — KEEP. Days lag1 0.684 stays KEEP. |


## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| c_ss_month leftover after days | **KEEP** | rank 0.635 OLS 0.635 fake_ols=False |
| 15-col Y3 card stem | **KEEP** | leftover after days rank 0.635 lives; beats size 0.617 by 0.077; not SIZE (ρ=0.342); not a twin (ρ days 0.424 salary 0.619). BETWEEN payer identity (ICC 0.990); not a days rewrite. Parent absorbs the card — do not edit it here. |
| Payroll twin vs c_salary_month | **no** | leftover after salary 0.668 R²=0.384 Jaccard=0.639 |
| Twin vs days / a_n_tx / tax | **no** | ρ days 0.424 salary 0.619 tax 0.330 |
| SIZE vs log1p(a_in3) | **no** | ρ=0.342 (gate ≥0.50); feature-report 0.362 |
| Inverse: days leftover after SS | **lives** | rank 0.641 OLS 0.643 |
| Q6 lag leftover after days_lag1 | **KEEP** | Y3 SS now 0.693 lag1 0.679 lag3 0.665; short lag1 0.686. Days lag1 0.684 (KEEP 0.684 CONFIRM). SS_lag1 leftover after days_lag1 0.631 lag3 0.606. Q6 KEEP. |
| Calendar | **monthly** | SS share Q-months 45.8% vs other 45.5% (ratio 1.01). Peak May 46.5%, trough Dec 44.6%. Shape **monthly** (tax was Q-peaked; salary monthly). Leftover after is_q_month 0.694. not a Q dummy. |
| Night quotes | **unchanged** | Y3 0.762/0.752 · days 0.711 · size 0.617 · salary 0.671 · TURNOVER 0.720/0.712 |


## 1. Coverage; modal; acf1; size ρ

Train c_ss_month cov 100.0% P(1) 45.7% modal 54.3% (feature-report 54.3% CONFIRM). acf1 0.600 (0.60 CONFIRM) acf3 0.144. ρ vs log1p(a_in3) 0.342 (quote 0.362 CONFIRM; not SIZE).

| slice | n_cm | n_co | cov | P(SS=1) | modal | n1 |
| --- | --- | --- | --- | --- | --- | --- |
| train all | 21,157 | 1214 | 100.0% | 45.7% | 54.3% | 9,665 |


## 2. Spearman twins

Spearman twins |ρ|≥0.80: none. vs days 0.424 vs a_n_tx 0.409 vs salary 0.619 vs missed_sal 0.004 vs tax 0.330 vs missed_tax 0.074. Jaccard(SS, salary) 0.639 both 7,075.

| vs | ρ | twin |
| --- | --- | --- |
| c_n_days_with_tx | 0.424 |  |
| a_n_tx | 0.409 |  |
| c_salary_month | 0.619 |  |
| c_missed_salary | 0.004 |  |
| c_tax_month | 0.330 |  |
| c_missed_tax | 0.074 |  |
| log1p(a_in3) | 0.342 |  |
| c_missed_ss | -0.120 |  |


## 3. Single-feature train group-fold AUROC

Y3 c_ss_month 0.693 (salary-qa 0.693 CONFIRM) vs days 0.711 (0.711 CONFIRM) vs size 0.617 (0.617 CONFIRM, Δ 0.077) vs salary_month 0.671 (0.671 CONFIRM) vs missed_sal 0.513 vs tax 0.611. Y2 SS 0.539. Loses to the 0.711 days bar.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | c_ss_month | 5,648 | 402 | 0.693 | 0.690 | 0.031 | -1 | 0.673 0.667 0.745 0.687 0.694 |
| y3_recover_cash_6m | c_salary_month | 5,648 | 402 | 0.671 | 0.672 | 0.050 | -1 | 0.611 0.659 0.745 0.689 0.652 |
| y3_recover_cash_6m | c_missed_salary | 5,648 | 402 | 0.513 | 0.512 | 0.010 | 1 | 0.531 0.510 0.507 0.507 0.508 |
| y3_recover_cash_6m | c_tax_month | 5,648 | 402 | 0.611 | 0.617 | 0.031 | -1 | 0.556 0.624 0.630 0.613 0.630 |
| y3_recover_cash_6m | c_missed_ss | 5,648 | 402 | 0.516 | 0.516 | 0.015 | 1 | 0.518 0.530 0.505 0.531 0.498 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | is_q_month | 5,648 | 402 | 0.495 | 0.503 | 0.012 | 1 | 0.514 0.487 0.494 0.498 0.483 |
| y3_recover_cash_6m | is_aug | 5,648 | 402 | 0.506 | 0.506 | 0.013 | 1 | 0.513 0.522 0.504 0.486 0.503 |
| y2_neg_2of3 | c_ss_month | 17,356 | 1,271 | 0.539 | 0.531 | 0.043 | -1 | 0.509 0.562 0.516 0.604 0.505 |
| y2_neg_2of3 | c_salary_month | 17,356 | 1,271 | 0.522 | 0.513 | 0.055 | 1 | 0.511 0.572 0.535 0.432 0.559 |
| y2_neg_2of3 | c_missed_salary | 17,356 | 1,271 | 0.494 | 0.503 | 0.009 | 1 | 0.508 0.485 0.495 0.486 0.496 |
| y2_neg_2of3 | c_tax_month | 17,356 | 1,271 | 0.535 | 0.541 | 0.026 | 1 | 0.559 0.558 0.540 0.520 0.498 |
| y2_neg_2of3 | c_missed_ss | 17,356 | 1,271 | 0.503 | 0.501 | 0.009 | -1 | 0.488 0.511 0.508 0.505 0.505 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.577 | 0.046 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 0.601 | 0.044 | 1 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | log1p(a_in3) | 14,968 | 1,044 | 0.552 | 0.540 | 0.046 | 1 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | is_q_month | 17,356 | 1,271 | 0.502 | 0.503 | 0.005 | -1 | 0.506 0.498 0.496 0.502 0.506 |
| y2_neg_2of3 | is_aug | 17,356 | 1,271 | 0.497 | 0.500 | 0.004 | 1 | 0.494 0.499 0.491 0.499 0.499 |


## 4. Honest leftover after days + inverse

Y3 leftover after days OLS 0.635 rank 0.635 ρ(resid,days)=-0.086 R²=0.178 (OLS and rank agree; resid is not a days clone). After salary_month rank 0.668 OLS 0.668 R²=0.384 (not a salary rewrite). Inverse: days leftover after SS rank 0.641 OLS 0.643 (0.711 bar lives). Honest leftover after days lives.

| y | control | OLS | rank | ρ(resid,ctrl) | R² | fake | honest_dies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | after days | 0.635 | 0.635 | -0.086 | 0.178 |  | no |
| y3_recover_cash_6m | after salary_month | 0.668 | 0.668 | 0.301 | 0.384 |  | no |
| y3_recover_cash_6m | after a_n_tx | 0.633 | 0.633 | -0.105 | 0.041 |  | no |
| y3_recover_cash_6m | after tax_month | 0.668 | 0.668 | -0.109 | 0.109 |  | no |
| y3_recover_cash_6m | after size | 0.665 | 0.665 | -0.172 | 0.124 |  | no |
| y3_recover_cash_6m | after days+salary | 0.633 | 0.633 | -0.016 | 0.416 |  | no |
| y2_neg_2of3 | after days | 0.588 | 0.588 | -0.086 | 0.178 |  | no |
| y2_neg_2of3 | after salary_month | 0.562 | 0.562 | 0.301 | 0.384 |  | no |
| y3_recover_cash_6m | days after SS (inverse) | 0.643 | 0.641 | 0.040 | 0.178 |  | no |


## 5. Leftover after salary_month

Y3 leftover after `c_salary_month` rank 0.668 OLS 0.668 R²=0.384. After days+salary 0.633. SS is not a salary rewrite.

## 6. SIZE terciles + ICC / demean

Y3 leftover after days T1 0.576 (lives); T2+T3 0.616 (lives).

| slice | n | n_pos | raw | leftover days | dies |
| --- | --- | --- | --- | --- | --- |
| T1 | 1,331 | 222 | 0.602 | 0.576 | no |
| T2+T3 | 4,317 | 180 | 0.663 | 0.616 | no |


c_ss_month ICC 0.990 (feature-report 0.99 CONFIRM BETWEEN). Y3 company-mean 0.720 demean 0.556 demean leftover after days 0.557.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | now | 5,648 | 402 | 0.693 | 0.690 | 0.031 | -1 | 0.673 0.667 0.745 0.687 0.694 |
| y3_recover_cash_6m | co_mean | 5,648 | 402 | 0.720 | 0.723 | 0.053 | -1 | 0.654 0.696 0.799 0.718 0.730 |
| y3_recover_cash_6m | demean | 5,648 | 402 | 0.556 | 0.547 | 0.052 | -1 | 0.639 0.574 0.516 0.532 0.519 |


## 7. Q6 — lag leftover after days_lag1

Y3 SS now 0.693 lag1 0.679 lag3 0.665; short lag1 0.686. Days lag1 0.684 (KEEP 0.684 CONFIRM). SS_lag1 leftover after days_lag1 0.631 lag3 0.606. Q6 KEEP.

| slice | col | n | n_pos | CV |
| --- | --- | --- | --- | --- |
| all | c_ss_month | 5,648 | 402 | 0.693 |
| all | c_ss_month_lag1 | 5,648 | 402 | 0.679 |
| all | c_ss_month_lag3 | 5,078 | 355 | 0.665 |
| all | c_n_days_with_tx | 5,648 | 402 | 0.711 |
| all | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 |
| short_<12 | c_ss_month | 3,723 | 252 | 0.693 |
| short_<12 | c_ss_month_lag1 | 3,723 | 252 | 0.686 |
| short_<12 | c_ss_month_lag3 | 3,153 | 205 | 0.676 |
| short_<12 | c_n_days_with_tx | 3,723 | 252 | 0.696 |
| short_<12 | c_n_days_with_tx_lag1 | 3,723 | 252 | 0.684 |


## 8. Calendar dummy

SS share Q-months 45.8% vs other 45.5% (ratio 1.01). Peak May 46.5%, trough Dec 44.6%. Shape **monthly** (tax was Q-peaked; salary monthly). Leftover after is_q_month 0.694. not a Q dummy.

| month | name | q | n | P(SS) |
| --- | --- | --- | --- | --- |
| 1 | Jan | 1 | 1,768 | 44.6% |
| 2 | Feb | 0 | 1,881 | 45.1% |
| 3 | Mar | 0 | 1,925 | 46.4% |
| 4 | Apr | 1 | 1,945 | 46.4% |
| 5 | May | 0 | 1,966 | 46.5% |
| 6 | Jun | 0 | 1,976 | 46.2% |
| 7 | Jul | 1 | 2,010 | 45.9% |
| 8 | Aug | 0 | 2,047 | 45.1% |
| 9 | Sep | 0 | 1,299 | 45.0% |
| 10 | Oct | 1 | 1,381 | 46.4% |
| 11 | Nov | 0 | 1,428 | 45.4% |
| 12 | Dec | 0 | 1,531 | 44.6% |


## 10. Dark 470 vs invoiced

Last-month ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). P(SS) invoiced 49.3% dark 39.3%. Y3 leftover invoiced 0.677 dark 0.574.

| slice | n_cm | P(SS) | Y3 raw | leftover days |
| --- | --- | --- | --- | --- |
| invoiced_744 | 13,554 | 49.3% | 0.731 | 0.677 |
| dark_470 | 7,603 | 39.3% | 0.647 | 0.574 |


## Extra — holdout coverage

Holdout coverage only (no fit): 72 co / 1,073 CM, cov 100.0% P(SS) 45.6%.

## Extra — fold-wise leftover

Y3 leftover-after-days rank folds 0.646 0.582 0.698 0.617 0.630 spread 0.116; OLS 0.646 0.582 0.698 0.617 0.630.

| fold | OLS | rank |
| --- | --- | --- |
| 0 | 0.646 | 0.646 |
| 1 | 0.582 | 0.582 |
| 2 | 0.698 | 0.698 |
| 3 | 0.617 | 0.617 |
| 4 | 0.630 | 0.630 |


## Extra — in-memory missed-SS (not a Y)

In-memory missed-SS (usual SS ≥3 in ≤6m and SS=0 this month; not a Y): prevalence 1.7%. Jaccard vs missed_salary 0.136. Y3 raw 0.516 leftover after days 0.677. Do not invent y_ss.

## Extra — SS-only vs salary-only

SS-only months 2,590 salary-only 1,409 either 11,074. Y3 leftover after salary+days 0.633.

Plot: `ss_qa.png`.

## What failed / next

- no replica miss; leftover after days lives (KEEP 0.635). Mixed leftover after days 0.419 dies — KEEP is BETWEEN payer identity, not a month flip.

## Extra — leftover after stacked controls

Y3 leftover after days+size 0.635; days+tax 0.636; days+size+salary 0.634; days+salary+tax 0.635; full stack days+size+salary+tax 0.635 (lives).

| control | OLS | rank | R² | dies |
| --- | --- | --- | --- | --- |
| after days+size | 0.634 | 0.635 | 0.196 | no |
| after days+tax | 0.636 | 0.636 | 0.207 | no |
| after days+ntx | 0.635 | 0.634 | 0.179 | no |
| after salary+tax | 0.653 | 0.653 | 0.404 | no |
| after days+size+salary | 0.633 | 0.634 | 0.421 | no |
| after days+salary+tax | 0.635 | 0.635 | 0.423 | no |
| after days+size+salary+tax | 0.634 | 0.635 | 0.427 | no |


## Extra — vs c_tax_month leftover

Jaccard(SS, tax) 0.486 both 6,744. Y3 tax raw 0.611. SS leftover after tax 0.668; tax leftover after SS 0.528; tax leftover after days 0.520; SS leftover after tax+days 0.636.

## Extra — BETWEEN trait vs within flip

BETWEEN leftover: co-mean SS after co-mean days 0.651; after co-mean salary 0.609. Within leftover: demean SS after demean days 0.571. Raw SS leftover after days + co-mean SS 0.566 (month flip after payer identity).

| spec | OLS | rank | R² | dies |
| --- | --- | --- | --- | --- |
| co_mean SS after co_mean days | 0.655 | 0.651 | 0.201 | no |
| co_mean SS after co_mean salary | 0.653 | 0.609 | 0.502 | no |
| co_mean SS after co_mean size | 0.652 | 0.646 | 0.168 | no |
| co_mean SS after co_mean days+salary | 0.629 | 0.582 | 0.519 | no |
| raw SS after co_mean SS | 0.554 | 0.554 | 0.854 | no |
| raw SS after days + co_mean SS | 0.536 | 0.566 | 0.857 | no |
| demean SS after demean days | 0.557 | 0.571 | 0.091 | no |
| demean SS after demean salary | 0.527 | 0.540 | 0.089 | YES |


## Extra — always / never / mixed cadence

Cadence always 348 / never 566 / mixed 300 companies. Y3 rate always 1.9% never 14.3% mixed 6.0%. Jaccard(SS, tax) 0.486.

| cadence | n_co | n_cm | n_y3 | Y3 rate | raw | leftover days |
| --- | --- | --- | --- | --- | --- | --- |
| always | 348 | 6049 | 2080 | 1.9% | LOW_POWER | — |
| never | 566 | 9653 | 1805 | 14.3% | 0.500 | 0.615 |
| mixed | 300 | 5455 | 1763 | 6.0% | 0.580 | 0.419 |


## Extra — SS≠salary leftover

Disagree months leftover after days 0.455; agree 0.666.

| slice | n | n_pos | raw | leftover days |
| --- | --- | --- | --- | --- |
| disagree SS≠salary | 1,177 | 56 | 0.455 | 0.455 |
| agree SS=salary | 4,471 | 346 | 0.717 | 0.666 |


## Extra — so-far / trail leftover

Leftover after days so_far<6 0.664; 6-11 0.632; ≥12 0.644; short 0.642; long —.

| slice | n | n_pos | raw | leftover days |
| --- | --- | --- | --- | --- |
| so_far<6 | 1,436 | 89 | 0.681 | 0.664 |
| so_far 6-11 | 2,287 | 163 | 0.704 | 0.632 |
| so_far≥12 | 1,925 | 150 | 0.688 | 0.644 |
| short_<12 | 3,723 | 252 | 0.693 | 0.642 |
| long_>=18 | 213 | 16 | LOW_POWER | — |


## Extra — drop chronic GROUP_0158/0172

Drop chronic ('GROUP_0158', 'GROUP_0172'): leftover after days 0.648 raw 0.701 (n_pos=402).

## Extra — fold leftover after salary / tax

Fold leftover after salary 0.669 0.635 0.710 0.649 0.677; after tax 0.661 0.631 0.726 0.657 0.664; after days+salary 0.653 0.585 0.681 0.608 0.640.

| fold | after salary | after tax | after days+salary |
| --- | --- | --- | --- |
| 0 | 0.669 | 0.661 | 0.653 |
| 1 | 0.635 | 0.631 | 0.585 |
| 2 | 0.710 | 0.726 | 0.681 |
| 3 | 0.649 | 0.657 | 0.608 |
| 4 | 0.677 | 0.664 | 0.640 |


## Extra — Q6 leftover after salary_lag1

SS_lag1 leftover after salary_lag1 0.655; after days_lag1+salary_lag1 0.626.

## Extra — inverse salary leftover after SS

Inverse: salary leftover after SS 0.626; after SS+days 0.591.

## Extra — in-memory SS share_6 (not a card col)

In-memory SS share_6 Y3 0.689 leftover after days 0.628 after c_ss_month 0.611. Not a card col.

## Extra — dark leftover after days+salary

Leftover after days+salary invoiced 0.671 dark 0.574.

| slice | leftover days+salary | OLS | dies |
| --- | --- | --- | --- |
| invoiced | 0.671 | 0.671 | no |
| dark | 0.574 | 0.574 | no |


## Extra — leftover after month dummies

Y3 leftover after 11 month dummies 0.700 R²=0.000.

## Extra — ever-SS / usual-SS (in-memory)

Ever-SS Y3 0.672 leftover after days 0.599 after salary 0.635. Usual-SS (≥50% months) Y3 0.687 leftover after days 0.623. Month SS leftover after ever-SS 0.686; after ever+days 0.637.

## Extra — mixed-cadence leftover (drops always/never)

Mixed-cadence only (n_pos=105): raw 0.580 leftover after days 0.419 after salary 0.549 after days+salary 0.433.

## Extra — SS-only vs salary-only companies

SS-only companies 55 sal-only 190 both 593 neither 376. SS-only leftover after days — raw —.

| type | n_co | n_y3 | Y3 rate |
| --- | --- | --- | --- |
| SS-only cos | 55 | 190 | 8.9% |
| sal-only cos | 190 | 787 | 8.4% |
| both | 593 | 3653 | 3.5% |
| neither | 376 | 1018 | 18.9% |


## Extra — Y3 rate cells days tercile × SS

Y3 rate cells by days tercile × SS (sign −: SS=1 should sit below SS=0 inside terciles).

| days tercile | SS | n | n_pos | Y3 rate |
| --- | --- | --- | --- | --- |
| low days | 0 | 1,267 | 218 | 17.2% |
| low days | 1 | 616 | 43 | 7.0% |
| mid | 0 | 557 | 60 | 10.8% |
| mid | 1 | 1,325 | 35 | 2.6% |
| high days | 0 | 441 | 25 | 5.7% |
| high days | 1 | 1,442 | 21 | 1.5% |


## Extra — Y2 leftover stack

Y2 leftover after days+salary 0.595; after tax 0.565.

## Extra — dark BETWEEN leftover

BETWEEN leftover after days-mean invoiced 0.711 dark 0.601.

| slice | BETWEEN leftover after days-mean |
| --- | --- |
| invoiced | 0.711 |
| dark | 0.601 |


## Extra — mixed leftover folds

Mixed leftover folds 0.285 0.463 0.491 0.502 0.354 spread 0.217; raw folds 0.740 0.590 0.487 0.577 0.506. n_pos total 105.

| fold | raw | leftover days | n_pos |
| --- | --- | --- | --- |
| 0 | 0.740 | 0.285 | 20 |
| 1 | 0.590 | 0.463 | 33 |
| 2 | 0.487 | 0.491 | 11 |
| 3 | 0.577 | 0.502 | 22 |
| 4 | 0.506 | 0.354 | 19 |


## Extra — company bootstrap leftover after days

Company bootstrap n=220 leftover-after-days rank p05/p50/p95 0.593 / 0.634 / 0.675; share <0.55 0.5%; wall 13s.

## Extra — leftover after other C stems

Leftover after other C stems: c_n_tx 0.633; c_recency_days 0.654; c_zero_in_month 0.669

| control | ρ | leftover | twin |
| --- | --- | --- | --- |
| c_n_tx | 0.409 | 0.633 |  |
| c_recency_days | -0.422 | 0.654 |  |
| c_zero_in_month | -0.267 | 0.669 |  |


## Extra — permutation null leftover after days

Within-fold permute SS leftover-after-days p05/p50/p95 0.586 / 0.609 / 0.631; observed 0.635; one-sided p(perm ≥ obs) 0.025; wall 10s.

## Extra — bootstrap leftover after salary / days+salary

Bootstrap leftover after salary p05/p50/p95 0.634 / 0.668 / 0.698; after days+salary 0.600 / 0.634 / 0.668; wall 23s.

## Extra — fold-wise residual leftover (no full-sample resid leak)

Fold-wise residual leftover after days 0.635 folds 0.646 0.582 0.698 0.617 0.630; after salary 0.668; after days+salary 0.633; inverse days after SS 0.644. lives vs chance 0.55.

## Extra — neither-payroll company dummy

Neither-payroll dummy Y3 0.657 leftover after days 0.562 after c_ss_month 0.593.

## Extra — leave-one-group leftover after days

Leave-one-group leftover after days (top 40 groups by Y3-labeled n): min/med/max 0.628 / 0.634 / 0.643.

| drop | n_left | leftover |
| --- | --- | --- |
| GROUP_0158 | 5309 | 0.636 |
| GROUP_0172 | 5459 | 0.643 |
| GROUP_0142 | 5464 | 0.633 |
| GROUP_0035 | 5479 | 0.634 |
| GROUP_0079 | 5496 | 0.640 |
| GROUP_0108 | 5510 | 0.637 |
| GROUP_0016 | 5512 | 0.631 |
| GROUP_0218 | 5521 | 0.632 |
| GROUP_0212 | 5531 | 0.631 |
| GROUP_0065 | 5533 | 0.638 |
| GROUP_0115 | 5545 | 0.639 |
| GROUP_0141 | 5545 | 0.632 |


## Extra — leftover after days by calendar month

Calendar-month leftover after days: Jan —, Feb 0.650, Mar —, Apr —, May —, Jun —, Jul —, Aug —, Sep —, Oct —, Nov —, Dec —; finite min 0.650

| month | n_pos | raw | leftover days |
| --- | --- | --- | --- |
| Jan | 45 | LOW_POWER | — |
| Feb | 53 | 0.697 | 0.650 |
| Mar | 22 | LOW_POWER | — |
| Apr | 20 | LOW_POWER | — |
| May | 22 | LOW_POWER | — |
| Jun | 28 | LOW_POWER | — |
| Jul | 37 | LOW_POWER | — |
| Aug | 30 | LOW_POWER | — |
| Sep | 24 | LOW_POWER | — |
| Oct | 33 | LOW_POWER | — |
| Nov | 44 | LOW_POWER | — |
| Dec | 44 | LOW_POWER | — |


## Extra — sibling / group-mean leftover

ρ(SS, group-mean SS) 0.593. Leftover after group-mean 0.680; after group-mean+days 0.612; co-mean SS after group-mean 0.682.

## Extra — BETWEEN leftover bootstrap

BETWEEN leftover bootstrap after days-mean p05/p50/p95 0.618 / 0.649 / 0.691; share<0.55 0.0%; wall 20s.


Elapsed 80s. Cuts: coverage/modal/acf/size, twins, singles (days/size/salary CONFIRM), leftover days + inverse, leftover salary, terciles, ICC, Q6 lag leftover, calendar, dark 470, holdout, folds, missed-SS, XOR.

Did **not**: rewrite `n_tx_qa.*` / `salary_qa.*` / `tax_qa.*` / `ops.py` / `gbm_core.py`, edit the 15-col card, grow TURNOVER, invent `y_ss`, merge I/M/J, rewrite `brief_map.md`, touch `product/`, write 0–100, fit holdout, run a new GBM, write the parent journal / LIVE / canvas.
