# Q3 zero-in — leftover after days, or SIZE / inverse-activity?

Generated `2026-09-19T04:31:37+02:00` by agent `87c77f5b`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_zero_in`. Do not revive `y6_zero_in_3`. Do not put zero-in on the 15-col card. Night Y3 quote stays 0.762 / 0.752. Days bar 0.711.

`c_zero_in_month` = 1 if no row with amount > 0 (any category). `c_zero_in_share_6` = mean of that flag over last ≤6 months (min_periods=1). Feature report: rare-event flag, modal 88.2%, size ρ −0.508 / −0.519.

## Headline

Flag pile **all-out (txs, no amount>0)** (empty 889 / all-out 1,597 of 2,486 zero-in CM). Modal 88.2% (CONFIRM 88.2%). SIZE month ρ=-0.429 share ρ=-0.500. Y3 month 0.580 share_6 0.617 vs size 0.617 (Δ -0.037) vs days 0.711. Leftover after days month 0.553 share 0.537. On the 44: **DROP from the 44**. PARK as health Y. Q6 **CLOSE**.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Not this flag. PARK as a health Y. Never-zero-in 761 / mostly-empty 91 / mostly-all-out 165. |
| 2 | Who is improving? | Not this table. |
| 3 | Who is turning? | **CLOSE** — empty vs all-out is all-out (txs, no amount>0). Leftover after days dies. SIZE YES. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | Inverse activity / size (Y6 failure mode), not a leftover quiet after days. |
| 6 | Months earlier? | **CLOSE** — contemporaneous Y3 0.580 loses to size / leftover dies; lag1 0.552 / lag3 0.551 have no leftover to lead. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `c_zero_in_month` on the 44 | **DROP from the 44** | SIZE (month vs a_in3 -0.429 vs |a_op_in| -0.508; share vs a_in3 -0.500 vs |a_op_in| -0.519); leftover after days dies (month 0.553 share 0.537 Δsize -0.063) |
| `c_zero_in_share_6` on the 44 | **DROP from the 44** | SIZE (month vs a_in3 -0.429 vs |a_op_in| -0.508; share vs a_in3 -0.500 vs |a_op_in| -0.519); leftover after days dies (month 0.553 share 0.537 Δsize -0.063) |
| either as a health Y | **PARK** | do not invent `y_zero_in`; do not revive Y6 |
| either on tonight's 15-col card | **no** | night engine stays days 0.711 / n_tx / salary |
| SIZE (\|ρ\| ≥ 0.50) | **YES** | KEEP clock log1p(a_in3) month -0.429 share -0.500; report clock log1p(\|a_op_in\|) month -0.508 share -0.519 |
| days / n_tx twin (\|ρ\|≥0.80) | **NO** | days -0.509 n_tx -0.514 |
| leftover after days | **dies** | month 0.553 share 0.537 Δsize -0.063 |
| Q6 lag1 / lag3 | **CLOSE** | Q6 Y3 month now 0.580 lag1 0.552 lag3 0.551; share_6 now 0.617 lag1 0.602 lag3 0.601; short now — lag1 —; long lag1 0.553. Short-book zero-in share 6.1% (181/2,970). **CLOSE** — contemporaneous Y3 0.580 loses to size / leftover dies; lag1 0.552 / lag3 0.551 have no leftover to lead. |
| Y6 `y6_zero_in_3` | **PARK / do not revive** | Y6 overlap n=17,083; Jaccard 0.286 Spearman 0.391 (share_6 0.378). P(Y6|zero-in now)=43.7% vs Y6 base 8.7%. Y3 using Y6-as-X 0.583. X does not leak y6_zero_in_3 (different window: now vs t+1..t+3). Do not revive Y6. Do not invent a zero-in Y. |

## 1. Prevalence + modal 88.2% + holdout coverage

Train `c_zero_in_month` prevalence 11.8% (n=2,486 / 21,157). Modal 0 share 88.2% (CONFIRM 88.2%). `c_zero_in_share_6` mean 0.116 modal 0.000 share 77.4%. Holdout coverage only: month mean 8.0% share mean 0.081 on 1,073 CM / 72 cos. No AUROC on holdout.

| split | col | n_cm | n_co | cov | mean | modal share |
| --- | --- | --- | --- | --- | --- | --- |
| train | c_zero_in_month | 21,157 | 1,214 | 100.0% | 11.8% | 88.2% |
| train | c_zero_in_share_6 | 21,157 | 1,214 | 100.0% | 0.116 | 77.4% |
| holdout | c_zero_in_month | 1,073 | 72 | 100.0% | 8.0% | 92.0% |
| holdout | c_zero_in_share_6 | 1,073 | 72 | 100.0% | 0.081 | 84.8% |


## 2. Empty grid vs all-out — which pile is the flag?

Train zero-in n=2,486 / 21,157 (11.8%). Empty grid 889 (4.2%; 35.8% of the flag). All-out 1,597 (7.5%; 64.2% of the flag). Empty ∩ has-in = 0 (must be 0). a_n_tx≠c_n_tx n=0. days==0 ≡ n_tx==0 agree 100.0%. Flag pile: **all-out (txs, no amount>0)**.

| pile | n_cm | share_cm | of_zero_in |
| --- | --- | --- | --- |
| has_in (amount>0) | 18671 | 88.2% | — |
| empty (n_tx=0) | 889 | 4.2% | 35.8% |
| all-out (n_tx>0, no amount>0) | 1597 | 7.5% | 64.2% |
| zero-in (empty ∪ all-out) | 2486 | 11.8% | 100% |
| days==0 | 889 | 4.2% | — |


Plot: `zero_in_piles.png`.

## 3. Formula vs raw txs (any amount>0)

Store `c_zero_in_month` vs raw (any amount>0) agree 100.0% (n_disagree=0). n_tx agree 100.0%. share_6 vs rolling-6 of store month max|Δ|=0.00e+00. All-out CM 1,597: only-neg 1,597 (100.0%), only-zero-amt 0. Not an ops.py bug.

## 4. Spearman vs days / n_tx / a_op_in / size / recency

Month vs log1p(a_in3) ρ=-0.429 not SIZE on the KEEP clock. Month vs log1p(|a_op_in|) ρ=-0.508 (CONFIRM −0.508) SIZE. share_6 vs log1p(a_in3) ρ=-0.500 SIZE. share_6 vs log1p(|a_op_in|) ρ=-0.519 (CONFIRM −0.519) SIZE. Month vs days -0.509 n_tx -0.514 a_op_in -0.506 recency 0.443. All-out vs days -0.356 size -0.315; empty vs size -0.272. not a |ρ|≥0.80 days/n_tx twin. not a recency twin.

| pair | ρ | twin |ρ|≥0.80 | SIZE |ρ|≥0.50 |
| --- | --- | --- | --- |
| c_zero_in_month vs c_n_days_with_tx | -0.509 |  |  |
| c_zero_in_month vs a_n_tx | -0.514 |  |  |
| c_zero_in_month vs c_n_tx | -0.514 |  |  |
| c_zero_in_month vs a_op_in | -0.506 |  |  |
| c_zero_in_month vs log_in3 | -0.429 |  |  |
| c_zero_in_month vs log_abs_opin | -0.508 |  | YES |
| c_zero_in_month vs c_recency_days | 0.443 |  |  |
| c_zero_in_month vs empty_month | 0.574 |  |  |
| c_zero_in_month vs all_out | 0.783 |  |  |
| c_zero_in_month vs zero_op_in | 0.729 |  |  |
| c_zero_in_share_6 vs c_n_days_with_tx | -0.583 |  |  |
| c_zero_in_share_6 vs a_n_tx | -0.588 |  |  |
| c_zero_in_share_6 vs a_op_in | -0.517 |  |  |
| c_zero_in_share_6 vs log_in3 | -0.500 |  | YES |
| c_zero_in_share_6 vs log_abs_opin | -0.519 |  | YES |
| c_zero_in_share_6 vs c_recency_days | 0.457 |  |  |
| c_zero_in_share_6 vs c_zero_in_month | 0.730 |  |  |
| empty_month vs c_n_days_with_tx | -0.348 |  |  |
| all_out vs c_n_days_with_tx | -0.356 |  |  |
| all_out vs a_n_tx | -0.363 |  |  |
| all_out vs log_in3 | -0.315 |  |  |
| empty_month vs log_in3 | -0.272 |  |  |
| zero_op_in vs c_zero_in_month | 0.729 |  |  |


## 5. Single-feature train group-fold AUROC

Y2 n=17,356 base 7.3%; Y3 stressed n=5,648 base 7.1%. Sign from the train side of each fold. Seed 20260918. Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica 0.711); size 0.617 (replica 0.617).

Y3 group-fold: month 0.580 share_6 0.617 empty 0.530 all-out 0.550 vs size 0.617 (quote 0.617, Δmonth -0.037) vs days 0.711 (night 0.711, replica OK) vs n_tx 0.703 vs a_op_in 0.676. Y2 month 0.522 share_6 0.530 vs size 0.552 days 0.571. Month beats size ≥0.02: NO.

| y | feature | n | n_pos | CV | sd | sign | train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | c_zero_in_month | 5,648 | 402 | 0.580 | 0.015 | 1 | 0.584 |
| y3_recover_cash_6m | c_zero_in_share_6 | 5,648 | 402 | 0.617 | 0.031 | 1 | 0.625 |
| y3_recover_cash_6m | empty_month | 5,648 | 402 | 0.530 | 0.018 | 1 | 0.529 |
| y3_recover_cash_6m | all_out | 5,648 | 402 | 0.550 | 0.025 | 1 | 0.554 |
| y3_recover_cash_6m | has_in | 5,648 | 402 | 0.580 | 0.015 | -1 | 0.584 |
| y3_recover_cash_6m | zero_op_in | 5,648 | 402 | 0.619 | 0.034 | 1 | 0.621 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.714 |
| y3_recover_cash_6m | a_op_in | 5,648 | 402 | 0.676 | 0.050 | -1 | 0.685 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 |
| y3_recover_cash_6m | c_recency_days | 5,648 | 402 | 0.659 | 0.051 | 1 | 0.665 |
| y2_neg_2of3 | c_zero_in_month | 17,356 | 1,271 | 0.522 | 0.027 | -1 | 0.521 |
| y2_neg_2of3 | c_zero_in_share_6 | 17,356 | 1,271 | 0.530 | 0.044 | -1 | 0.531 |
| y2_neg_2of3 | empty_month | 17,356 | 1,271 | 0.504 | 0.014 | -1 | 0.504 |
| y2_neg_2of3 | all_out | 17,356 | 1,271 | 0.518 | 0.016 | -1 | 0.517 |
| y2_neg_2of3 | has_in | 17,356 | 1,271 | 0.522 | 0.027 | 1 | 0.521 |
| y2_neg_2of3 | zero_op_in | 17,356 | 1,271 | 0.500 | 0.044 | -1 | 0.513 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.046 | 1 | 0.577 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 0.044 | 1 | 0.601 |
| y2_neg_2of3 | a_op_in | 17,356 | 1,271 | 0.541 | 0.047 | 1 | 0.532 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 |
| y2_neg_2of3 | c_recency_days | 17,356 | 1,271 | 0.546 | 0.039 | -1 | 0.541 |


KEEP-as-X on the 44: leftover after days beats size by ≥0.02 **and** not SIZE (|ρ|≥0.5). Still do not put it on tonight's 15-col card.

## 6. Residual after days and after `a_n_tx`

Month leftover after days 0.553 / after n_tx 0.544 / card 0.556 (Δsize -0.063). share_6 leftover after days 0.537 (Δsize -0.079). All-out leftover after days 0.614. Among n_tx>0: all-out Y3 0.554 vs size 0.612 vs days 0.699; month-on-busy 0.554; all-out resid-days on busy 0.593. ρ(month-resid, days)=0.427. Leftover dies — CLOSE as quiet twin of days/n_tx.

| y | feature | n | n_pos | CV | sign | Δsize | slope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | z_resid_days | 5,648 | 402 | 0.553 | -1 | -0.063 | -0.018 |
| y3_recover_cash_6m | z_resid_ntx | 5,648 | 402 | 0.544 | -1 | -0.073 | -0.000 |
| y3_recover_cash_6m | z_resid_card | 5,648 | 402 | 0.556 | -1 | -0.061 | -0.020 |
| y3_recover_cash_6m | z_resid_opin | 5,648 | 402 | 0.520 | -1 | -0.097 | -0.000 |
| y3_recover_cash_6m | sh_resid_days | 5,648 | 402 | 0.537 | -1 | -0.079 | -0.016 |
| y3_recover_cash_6m | sh_resid_ntx | 5,648 | 402 | 0.540 | 1 | -0.077 | -0.000 |
| y3_recover_cash_6m | sh_resid_card | 5,648 | 402 | 0.542 | -1 | -0.074 | -0.017 |
| y3_recover_cash_6m | ao_resid_days | 5,648 | 402 | 0.614 | -1 | -0.003 | -0.010 |
| y3_recover_cash_6m | ao_resid_ntx | 5,648 | 402 | 0.604 | -1 | -0.012 | -0.000 |
| y3_recover_cash_6m | em_resid_days | 5,648 | 402 | 0.651 | -1 | 0.034 | -0.007 |
| y3_recover_cash_6m | sh_resid_z | 5,648 | 402 | 0.519 | 1 | -0.097 | 0.646 |
| y2_neg_2of3 | z_resid_days | 17,356 | 1,271 | 0.530 | 1 | -0.022 | -0.018 |
| y2_neg_2of3 | z_resid_ntx | 17,356 | 1,271 | 0.555 | 1 | 0.003 | -0.000 |
| y2_neg_2of3 | z_resid_card | 17,356 | 1,271 | 0.527 | 1 | -0.025 | -0.020 |
| y2_neg_2of3 | z_resid_opin | 17,356 | 1,271 | 0.494 | -1 | -0.058 | -0.000 |
| y2_neg_2of3 | sh_resid_days | 17,356 | 1,271 | 0.523 | 1 | -0.028 | -0.016 |
| y2_neg_2of3 | sh_resid_ntx | 17,356 | 1,271 | 0.526 | 1 | -0.026 | -0.000 |
| y2_neg_2of3 | sh_resid_card | 17,356 | 1,271 | 0.521 | 1 | -0.031 | -0.017 |
| y2_neg_2of3 | ao_resid_days | 17,356 | 1,271 | 0.538 | 1 | -0.014 | -0.010 |
| y2_neg_2of3 | ao_resid_ntx | 17,356 | 1,271 | 0.563 | 1 | 0.012 | -0.000 |
| y2_neg_2of3 | em_resid_days | 17,356 | 1,271 | 0.564 | 1 | 0.012 | -0.007 |
| y2_neg_2of3 | sh_resid_z | 17,356 | 1,271 | 0.514 | -1 | -0.038 | 0.646 |


## 7. SIZE terciles

Y3 month CV inside size terciles that beat size ≥0.02: ['T1']. T1 (small) 0.573; T3 (large) 0.534. If skill is only T1, it is the inverse-size / Y6 failure mode.

| tercile | feature | n | n_pos | zero_in | empty | all_out | CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | c_zero_in_month | 1,572 | 269 | 26.5% | 10.4% | 16.1% | 0.573 |
| T1 | c_zero_in_share_6 | 1,572 | 269 | 26.5% | 10.4% | 16.1% | 0.588 |
| T1 | empty_month | 1,572 | 269 | 26.5% | 10.4% | 16.1% | 0.525 |
| T1 | all_out | 1,572 | 269 | 26.5% | 10.4% | 16.1% | 0.549 |
| T1 | log_in3 | 1,533 | 261 | 26.5% | 10.4% | 16.1% | 0.427 |
| T1 | c_n_days_with_tx | 1,572 | 269 | 26.5% | 10.4% | 16.1% | 0.603 |
| T2 | c_zero_in_month | 2,044 | 73 | 4.7% | 0.9% | 3.8% | 0.527 |
| T2 | c_zero_in_share_6 | 2,044 | 73 | 4.7% | 0.9% | 3.8% | 0.544 |
| T2 | empty_month | 2,044 | 73 | 4.7% | 0.9% | 3.8% | 0.487 |
| T2 | all_out | 2,044 | 73 | 4.7% | 0.9% | 3.8% | 0.519 |
| T2 | log_in3 | 2,006 | 72 | 4.7% | 0.9% | 3.8% | 0.634 |
| T2 | c_n_days_with_tx | 2,044 | 73 | 4.7% | 0.9% | 3.8% | 0.673 |
| T3 | c_zero_in_month | 2,032 | 60 | 2.8% | 0.8% | 2.0% | 0.534 |
| T3 | c_zero_in_share_6 | 2,032 | 60 | 2.8% | 0.8% | 2.0% | 0.557 |
| T3 | empty_month | 2,032 | 60 | 2.8% | 0.8% | 2.0% | 0.524 |
| T3 | all_out | 2,032 | 60 | 2.8% | 0.8% | 2.0% | 0.483 |
| T3 | log_in3 | 1,989 | 58 | 2.8% | 0.8% | 2.0% | 0.620 |
| T3 | c_n_days_with_tx | 2,032 | 60 | 2.8% | 0.8% | 2.0% | 0.733 |


## 8. vs `y6_zero_in_3` (do not revive)

Y6 overlap n=17,083; Jaccard 0.286 Spearman 0.391 (share_6 0.378). P(Y6|zero-in now)=43.7% vs Y6 base 8.7%. Y3 using Y6-as-X 0.583. X does not leak y6_zero_in_3 (different window: now vs t+1..t+3). Do not revive Y6. Do not invent a zero-in Y.

| item | value |
| --- | --- |
| overlap n (Y6 defined) | 17,083 |
| c_zero_in_month=1 on overlap | 1,546 |
| y6_zero_in_3=1 on overlap | 1,490 |
| intersection | 675 |
| Jaccard | 0.286 |
| Spearman month | 0.391 |
| Spearman share_6 | 0.378 |
| P(Y6|zero-in now) | 43.7% |
| P(zero-in now|Y6) | 45.3% |
| Y6 base on overlap | 8.7% |


## 9. Dark 470 vs invoiced 744

Train last-month: ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Mean company zero-in CM: invoiced 10.9% vs dark 11.0%. Same bank-book zero-in rate. Y3 month CV invoiced 0.584 dark 0.579. Holdout ever-ERP coverage only: 40/72.

| group | n_co | ever_zero | zero_cm | empty_cm | all_out_cm | share6 |
| --- | --- | --- | --- | --- | --- | --- |
| ever_erp_744 | 744 | 38.3% | 10.9% | 3.3% | 7.6% | 0.112 |
| never_erp_470 | 470 | 35.7% | 11.0% | 4.6% | 6.5% | 0.108 |


| group | n_cm | n_co | zero | empty | all_out |
| --- | --- | --- | --- | --- | --- |
| ever_erp | 13,554 | 744 | 11.6% | 3.6% | 8.0% |
| never_erp | 7,603 | 470 | 12.0% | 5.2% | 6.8% |


## 10. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)

Chronic 12 names (0158/0172, ≥50% labeled months below 0): 12. Y2 month CV full 0.522 → drop-12 0.520. Zero-in share on 12 1.6% vs rest 11.9%. Drop does not flip Y2 (≥0.03).

| y | feature | slice | n | n_pos | CV |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | c_zero_in_month | full | 17,356 | 1,271 | 0.522 |
| y2_neg_2of3 | c_zero_in_month | drop_12 | 17,140 | 1,094 | 0.520 |
| y2_neg_2of3 | c_zero_in_share_6 | full | 17,356 | 1,271 | 0.530 |
| y2_neg_2of3 | c_zero_in_share_6 | drop_12 | 17,140 | 1,094 | 0.521 |
| y2_neg_2of3 | log1p_a_in3 | full | 14,968 | 1,044 | 0.552 |
| y2_neg_2of3 | log1p_a_in3 | drop_12 | 14,776 | 881 | 0.545 |
| y2_neg_2of3 | c_n_days_with_tx | full | 17,356 | 1,271 | 0.571 |
| y2_neg_2of3 | c_n_days_with_tx | drop_12 | 17,140 | 1,094 | 0.549 |
| y3_recover_cash_6m | c_zero_in_month | full | 5,648 | 402 | 0.580 |
| y3_recover_cash_6m | c_zero_in_month | drop_12 | 5,486 | 402 | 0.580 |
| y3_recover_cash_6m | c_zero_in_share_6 | full | 5,648 | 402 | 0.617 |
| y3_recover_cash_6m | c_zero_in_share_6 | drop_12 | 5,486 | 402 | 0.616 |


## 11. ICC / company-demean (share_6 acf1 0.87)

share_6 ICC=0.973 acf1=0.866 (CONFIRM 0.87). month ICC=0.933 acf1=-0.027. Y3 share_6 raw 0.617 vs demean 0.648 (drop -0.031). TRAIT (BETWEEN company style).

| col | ICC | acf1 | acf3 | Y3 raw | Y3 demean | drop |
| --- | --- | --- | --- | --- | --- | --- |
| c_zero_in_month | 0.933 | -0.027 | -0.000 | 0.580 | 0.597 | -0.017 |
| c_zero_in_share_6 | 0.973 | 0.866 | 0.602 | 0.617 | 0.648 | -0.031 |
| empty_month | 0.902 | 0.357 | 0.234 | 0.530 | 0.589 | -0.059 |
| all_out | 0.913 | -0.045 | -0.050 | 0.550 | 0.597 | -0.047 |


## 12. Q6 — lag1 / lag3 on short vs long books

Q6 Y3 month now 0.580 lag1 0.552 lag3 0.551; share_6 now 0.617 lag1 0.602 lag3 0.601; short now — lag1 —; long lag1 0.553. Short-book zero-in share 6.1% (181/2,970). **CLOSE** — contemporaneous Y3 0.580 loses to size / leftover dies; lag1 0.552 / lag3 0.551 have no leftover to lead.

| y | slice | col | n | n_pos | x_mean | CV |
| --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | all | c_zero_in_month | 17,356 | 1,271 | 11.3% | 0.522 |
| y2_neg_2of3 | all | c_zero_in_month_lag1 | 16,161 | 1,151 | 11.2% | 0.521 |
| y2_neg_2of3 | all | c_zero_in_month_lag3 | 13,776 | 938 | 11.3% | 0.517 |
| y2_neg_2of3 | all | c_zero_in_share_6 | 17,356 | 1,271 | 0.116 | 0.530 |
| y2_neg_2of3 | all | c_zero_in_share_6_lag1 | 16,161 | 1,151 | 0.116 | 0.530 |
| y2_neg_2of3 | all | c_zero_in_share_6_lag3 | 13,776 | 938 | 0.117 | 0.534 |
| y2_neg_2of3 | short_<12 | c_zero_in_month | 1,833 | 147 | 5.8% | 0.486 |
| y2_neg_2of3 | short_<12 | c_zero_in_month_lag1 | 1,497 | 108 | 5.9% | 0.483 |
| y2_neg_2of3 | short_<12 | c_zero_in_month_lag3 | 830 | 46 | 8.6% | LOW_POWER |
| y2_neg_2of3 | short_<12 | c_zero_in_share_6 | 1,833 | 147 | 0.081 | 0.522 |
| y2_neg_2of3 | short_<12 | c_zero_in_share_6_lag1 | 1,497 | 108 | 0.086 | 0.523 |
| y2_neg_2of3 | short_<12 | c_zero_in_share_6_lag3 | 830 | 46 | 0.104 | LOW_POWER |
| y2_neg_2of3 | long_>=12 | c_zero_in_month | 15,523 | 1,124 | 11.9% | 0.524 |
| y2_neg_2of3 | long_>=12 | c_zero_in_month_lag1 | 14,664 | 1,043 | 11.8% | 0.523 |
| y2_neg_2of3 | long_>=12 | c_zero_in_month_lag3 | 12,946 | 892 | 11.5% | 0.516 |
| y2_neg_2of3 | long_>=12 | c_zero_in_share_6 | 15,523 | 1,124 | 0.120 | 0.529 |
| y2_neg_2of3 | long_>=12 | c_zero_in_share_6_lag1 | 14,664 | 1,043 | 0.119 | 0.529 |
| y2_neg_2of3 | long_>=12 | c_zero_in_share_6_lag3 | 12,946 | 892 | 0.118 | 0.531 |
| y2_neg_2of3 | so_far>=4 | c_zero_in_month | 13,776 | 938 | 11.1% | 0.529 |
| y2_neg_2of3 | so_far>=4 | c_zero_in_month_lag1 | 13,776 | 938 | 10.9% | 0.528 |
| y2_neg_2of3 | so_far>=4 | c_zero_in_month_lag3 | 13,776 | 938 | 11.3% | 0.517 |
| y2_neg_2of3 | so_far>=4 | c_zero_in_share_6 | 13,776 | 938 | 0.111 | 0.539 |
| y2_neg_2of3 | so_far>=4 | c_zero_in_share_6_lag1 | 13,776 | 938 | 0.112 | 0.537 |
| y2_neg_2of3 | so_far>=4 | c_zero_in_share_6_lag3 | 13,776 | 938 | 0.117 | 0.534 |
| y3_recover_cash_6m | all | c_zero_in_month | 5,648 | 402 | 5.5% | 0.580 |
| y3_recover_cash_6m | all | c_zero_in_month_lag1 | 5,648 | 402 | 5.4% | 0.552 |
| y3_recover_cash_6m | all | c_zero_in_month_lag3 | 5,078 | 355 | 5.8% | 0.551 |
| y3_recover_cash_6m | all | c_zero_in_share_6 | 5,648 | 402 | 0.059 | 0.617 |
| y3_recover_cash_6m | all | c_zero_in_share_6_lag1 | 5,648 | 402 | 0.061 | 0.602 |
| y3_recover_cash_6m | all | c_zero_in_share_6_lag3 | 5,078 | 355 | 0.067 | 0.601 |
| y3_recover_cash_6m | short_<12 | c_zero_in_month | 146 | 13 | 4.1% | LOW_POWER |
| y3_recover_cash_6m | short_<12 | c_zero_in_month_lag1 | 146 | 13 | 4.8% | LOW_POWER |
| y3_recover_cash_6m | short_<12 | c_zero_in_month_lag3 | 53 | 1 | 9.4% | LOW_POWER |
| y3_recover_cash_6m | short_<12 | c_zero_in_share_6 | 146 | 13 | 0.063 | LOW_POWER |
| y3_recover_cash_6m | short_<12 | c_zero_in_share_6_lag1 | 146 | 13 | 0.076 | LOW_POWER |
| y3_recover_cash_6m | short_<12 | c_zero_in_share_6_lag3 | 53 | 1 | 0.094 | LOW_POWER |
| y3_recover_cash_6m | long_>=12 | c_zero_in_month | 5,502 | 389 | 5.6% | 0.582 |
| y3_recover_cash_6m | long_>=12 | c_zero_in_month_lag1 | 5,502 | 389 | 5.4% | 0.553 |
| y3_recover_cash_6m | long_>=12 | c_zero_in_month_lag3 | 5,025 | 354 | 5.8% | 0.551 |
| y3_recover_cash_6m | long_>=12 | c_zero_in_share_6 | 5,502 | 389 | 0.059 | 0.621 |
| y3_recover_cash_6m | long_>=12 | c_zero_in_share_6_lag1 | 5,502 | 389 | 0.061 | 0.606 |
| y3_recover_cash_6m | long_>=12 | c_zero_in_share_6_lag3 | 5,025 | 354 | 0.066 | 0.601 |
| y3_recover_cash_6m | so_far>=4 | c_zero_in_month | 5,078 | 355 | 5.4% | 0.584 |
| y3_recover_cash_6m | so_far>=4 | c_zero_in_month_lag1 | 5,078 | 355 | 5.1% | 0.556 |
| y3_recover_cash_6m | so_far>=4 | c_zero_in_month_lag3 | 5,078 | 355 | 5.8% | 0.551 |
| y3_recover_cash_6m | so_far>=4 | c_zero_in_share_6 | 5,078 | 355 | 0.057 | 0.623 |
| y3_recover_cash_6m | so_far>=4 | c_zero_in_share_6_lag1 | 5,078 | 355 | 0.059 | 0.610 |
| y3_recover_cash_6m | so_far>=4 | c_zero_in_share_6_lag3 | 5,078 | 355 | 0.067 | 0.601 |


## 13. amount>0 vs `a_op_in==0` (CAT_MAP hole)

amount>0 ∩ a_op_in==0 (transfer/invest-in, not CAT_MAP op_in): 1,725 (8.2%). zero-in ∩ a_op_in>0 = 10 (must be 0). Y3 of that hole flag 0.539. Zero-in is any-category inflow, not the op_in bucket Y6 used.

| cell | n_cm | share |
| --- | --- | --- |
| zero-in ∩ a_op_in==0 | 2476 | 11.7% |
| zero-in ∩ a_op_in>0 | 10 | 0.0% |
| has amount>0 ∩ a_op_in==0 | 1725 | 8.2% |
| has amount>0 ∩ a_op_in>0 | 16946 | 80.1% |


## 14. Calendar of empty vs all-out

Zero-in peak Aug 15.5%, trough Oct 10.7%. August 15.5%. Not a strong calendar dummy.

| month | n_cm | zero_in | empty | all_out |
| --- | --- | --- | --- | --- |
| Jan | 1,768 | 11.0% | 2.8% | 8.1% |
| Feb | 1,881 | 11.4% | 3.8% | 7.6% |
| Mar | 1,925 | 10.9% | 4.1% | 6.8% |
| Apr | 1,945 | 10.8% | 3.7% | 7.1% |
| May | 1,966 | 11.0% | 4.6% | 6.5% |
| Jun | 1,976 | 12.1% | 5.4% | 6.8% |
| Jul | 2,010 | 11.7% | 5.0% | 6.8% |
| Aug | 2,047 | 15.5% | 7.8% | 7.7% |
| Sep | 1,299 | 12.6% | 2.0% | 10.6% |
| Oct | 1,381 | 10.7% | 2.6% | 8.1% |
| Nov | 1,428 | 11.4% | 3.4% | 8.1% |
| Dec | 1,531 | 11.4% | 3.5% | 7.9% |


## 15. Holdout coverage only (no AUROC)

Holdout 72 coverage only: 1,073 CM / 72 cos. zero-in mean 8.0%; empty 2.7%; all-out 5.3%. No AUROC claim.

| col | n_cm | n_co | cov | mean |
| --- | --- | --- | --- | --- |
| c_zero_in_month | 1073 | 72 | 100.0% | 8.0% |
| c_zero_in_share_6 | 1073 | 72 | 100.0% | 0.081 |
| empty_month | 1073 | 72 | 100.0% | 2.7% |
| all_out | 1073 | 72 | 100.0% | 5.3% |


## 16. Y rates on empty / all-out / has-in

Y3 rate: empty 26.8% (n_lab=112) vs all-out 27.4% vs has-in 5.9%. A leftover Q3 going-quiet would show all-out ≠ empty. Inverse-activity would pile both.

| y | slice | n_cm | n_lab | n_pos | rate |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | has_in | 18,671 | 15,398 | 1,176 | 7.6% |
| y2_neg_2of3 | empty | 889 | 592 | 35 | 5.9% |
| y2_neg_2of3 | all_out | 1,597 | 1,366 | 60 | 4.4% |
| y2_neg_2of3 | zero_in | 2,486 | 1,958 | 95 | 4.9% |
| y3_recover_cash_6m | has_in | 18,671 | 5,335 | 317 | 5.9% |
| y3_recover_cash_6m | empty | 889 | 112 | 30 | 26.8% |
| y3_recover_cash_6m | all_out | 1,597 | 201 | 55 | 27.4% |
| y3_recover_cash_6m | zero_in | 2,486 | 313 | 85 | 27.2% |


## 17. All-out category mix (raw txs)

All-out months: 1,597 CM, 8,202 txs. Top token `uncategorized`. If the pile is payroll/tax/fee outflows only, it is category-all-out, not an empty book.

| category | n_tx | share_tx | n_cm | net_amt |
| --- | --- | --- | --- | --- |
| uncategorized | 2898 | 35.3% | 620 | -447,808,579 |
| payment | 1747 | 21.3% | 630 | -150,389,267 |
| utility | 1200 | 14.6% | 494 | -3,844,429 |
| fee | 614 | 7.5% | 408 | -604,893 |
| tax | 522 | 6.4% | 348 | -2,748,854 |
| social_security | 291 | 3.5% | 231 | -801,475 |
| interest_charge | 267 | 3.3% | 255 | -57,203 |
| salary | 254 | 3.1% | 177 | -2,774,092 |
| bulk_payment | 162 | 2.0% | 105 | -1,843,731 |
| debt_repayment | 82 | 1.0% | 63 | -1,059,163 |
| cash_withdrawal | 49 | 0.6% | 43 | -65,056 |
| pos_withdrawal | 44 | 0.5% | 36 | -11,115 |


## 18. Ever-zero-in companies

Train companies: never zero-in 761, mostly-empty 91, mostly-all-out 165, rare 197 / 1214.

| kind | n_co | share_co | z_rate_p50 | size_p50 |
| --- | --- | --- | --- | --- |
| mostly_allout | 165 | 13.6% | 0.381 | 9.081 |
| mostly_empty | 91 | 7.5% | 0.455 | 8.321 |
| never | 761 | 62.7% | 0.000 | 13.354 |
| rare | 197 | 16.2% | 0.083 | 11.969 |


## 19. share_6 leftover after the month flag

Y3 share_6 0.617 after the month flag 0.519 (slope 0.646). share_6 is the month flag smoothed — no extra trail.

| y | share_6 | month | share_after_month | n_pos |
| --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 0.617 | 0.580 | 0.519 | 402 |
| y2_neg_2of3 | 0.530 | 0.522 | 0.514 | 1,271 |


## 20. Stressed / Y3-labeled population (the card)

Y3-labeled (stressed) CM 5,648: zero-in 313 empty 112 all-out 201. This is the card population. Zero-in is not on the 15-col card.

| feature | n | n_pos | mean | CV |
| --- | --- | --- | --- | --- |
| c_zero_in_month | 5,648 | 402 | 0.055 | 0.580 |
| c_zero_in_share_6 | 5,648 | 402 | 0.059 | 0.617 |
| empty_month | 5,648 | 402 | 0.020 | 0.530 |
| all_out | 5,648 | 402 | 0.036 | 0.550 |
| c_n_days_with_tx | 5,648 | 402 | 16.426 | 0.711 |
| a_n_tx | 5,648 | 402 | 138.580 | 0.703 |
| log_in3 | 5,528 | 391 | 12.516 | 0.617 |


## 21. Busy-only leftover (n_tx>0)

Busy-only (n_tx>0) Y3: all-out 0.554 vs size 0.612 (Δ -0.058) vs days 0.699. All-out on busy months does not beat size — not a leftover Q3.

| y | feature | n | n_pos | CV |
| --- | --- | --- | --- | --- |
| y3_recover_cash_6m | all_out | 5,536 | 372 | 0.554 |
| y3_recover_cash_6m | c_zero_in_month | 5,536 | 372 | 0.554 |
| y3_recover_cash_6m | log1p_a_in3 | 5,422 | 361 | 0.612 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,536 | 372 | 0.699 |
| y3_recover_cash_6m | a_n_tx | 5,536 | 372 | 0.691 |
| y3_recover_cash_6m | a_op_in | 5,536 | 372 | 0.663 |
| y2_neg_2of3 | all_out | 16,764 | 1,236 | 0.519 |
| y2_neg_2of3 | c_zero_in_month | 16,764 | 1,236 | 0.519 |
| y2_neg_2of3 | log1p_a_in3 | 14,417 | 1,015 | 0.550 |
| y2_neg_2of3 | c_n_days_with_tx | 16,764 | 1,236 | 0.571 |
| y2_neg_2of3 | a_n_tx | 16,764 | 1,236 | 0.600 |
| y2_neg_2of3 | a_op_in | 16,764 | 1,236 | 0.539 |


## 22. Feature-report SIZE clock `log1p(|a_op_in|)`

Y3 log1p(|a_op_in|) 0.677 (feature-report SIZE clock). Month leftover after that clock 0.520; share_6 leftover 0.526; month after days+opin 0.541. If leftover dies on the report clock too, the −0.508 flag is the same inverse-activity story.

| feature | n | n_pos | CV |
| --- | --- | --- | --- |
| log1p_|a_op_in| | 5,648 | 402 | 0.677 |
| z_resid_opin_sz | 5,648 | 402 | 0.520 |
| sh_resid_opin_sz | 5,648 | 402 | 0.526 |
| z_resid_days_opin | 5,648 | 402 | 0.541 |


## 23. The 10-row hole — zero-in ∩ `a_op_in>0`

zero-in ∩ a_op_in≠0: 10 train CM. a_op_in p50=-802.68 max=-18.78; empty among them 0. Signed CAT_MAP op_in without any amount>0 (refunds / dust). Do not patch ops.py — store vs raw amount>0 already agreed 100%.

| category | n_tx | net | max_amt |
| --- | --- | --- | --- |
| uncategorized | 15 | -49,952.79 | -19.74 |
| collection_refund | 11 | -230,707.72 | -18.78 |
| utility | 10 | -79,291.84 | -3.15 |
| payment | 9 | -133,193.96 | -319.71 |
| tax | 6 | -16,706.14 | -0.60 |
| bulk_payment | 5 | -270,479.92 | -1,428.35 |
| fee | 2 | -3.20 | -0.45 |
| social_security | 2 | -4,655.83 | -722.54 |


## 24. T1 leftover after days

T1-only Y3: month 0.573 leftover-days 0.538 vs size 0.427 (Δ 0.111) vs days 0.603. T1 leftover loses to days (size clock is inverted inside T1) — still inverse-activity.

| feature | n | n_pos | CV |
| --- | --- | --- | --- |
| c_zero_in_month | 1,572 | 269 | 0.573 |
| c_zero_in_share_6 | 1,572 | 269 | 0.588 |
| log1p_a_in3 | 1,533 | 261 | 0.427 |
| c_n_days_with_tx | 1,572 | 269 | 0.603 |
| z_resid_days | 1,572 | 269 | 0.538 |
| sh_resid_days | 1,572 | 269 | 0.514 |
| empty_month | 1,572 | 269 | 0.525 |
| all_out | 1,572 | 269 | 0.549 |


## 25. Y6 vs size (inverse-size failure mode)

Y6 vs size 0.854 vs log1p(|a_op_in|) 0.878 (CONFIRM inverse-size / Y6 failure mode). P(Y6|zero-in)=43.7% empty 62.9% all-out 41.2%. Do not revive Y6.

| feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- |
| log1p_a_in3 | 14,698 | 1,300 | 0.854 | -1 |
| log1p_|a_op_in| | 17,083 | 1,490 | 0.878 | -1 |
| c_n_days_with_tx | 17,083 | 1,490 | 0.865 | -1 |
| c_zero_in_month | 17,083 | 1,490 | 0.694 | 1 |
| empty_month | 17,083 | 1,490 | 0.535 | 1 |
| all_out | 17,083 | 1,490 | 0.659 | 1 |


## 26. All-out resid 0.614 — fake leftover?

All-out OLS leftover after days Y3 0.614 ρ(resid, days)=0.664 slope=-0.010. FAKE leftover — resid still tracks days. Busy-only all-out 0.554 is the honest number.

## 27. share_6 leftover after days + size

share_6 leftover after days+size 0.494; month after days+size 0.539; demeaned share after days 0.686 (raw demean 0.648). Double residual dies — quiet twin + SIZE, nothing left.

| feature | n | n_pos | CV |
| --- | --- | --- | --- |
| sh_resid_days_size | 5,528 | 391 | 0.494 |
| z_resid_days_size | 5,528 | 391 | 0.539 |
| sh_demean | 5,648 | 402 | 0.648 |
| sh_demean_resid_days | 5,648 | 402 | 0.686 |


## 28. Demean leftover honesty + uncat overlap

Demeaned share_6 after days 0.686 ρ(resid,days)=0.332 (resid orthogonal to days). All-out vs uncat ρ=-0.122 (not an uncat twin); uncat share all-out 24.4% vs has-in 25.9%. Demean leftover would KEEP as a transform — still not the stored column, not on the 44.

| item | value |
| --- | --- |
| Y3 demean | 0.648 |
| Y3 demean after days | 0.686 |
| ρ(demean-resid, days) | 0.332 |
| ρ(demean, log1p(a_in3)) | -0.094 |
| slope vs days | -0.003 |
| ρ(all_out, a_uncat_share) | -0.122 |
| ρ(zero-in, a_uncat_share) | -0.122 |
| mean uncat on all-out | 24.4% |
| mean uncat on has-in | 25.9% |


## What we did not do

- Did not edit `ops.py`, `y6_activity.py`, recency/gap_sd/transfer QA, `product/`, parquet / duckdb.
- Did not run `python -m analysis.targets.build_targets`.
- Did not write the parent journal, a 0–100 formula, or a new Y.
- Did not revive `y6_zero_in_3` or put zero-in on the 15-col card.
- Did not change the night Y3 quote 0.762 / 0.752 or days 0.711.

## What failed / next (held for wave note)

- zero-in ∩ a_op_in>0 n=10 (CAT_MAP hole, not an ops.py amount>0 bug)

Elapsed 9s. Must-do 1–12 plus opin-hole, calendar, holdout, Y rates, all-out mix, ever-kind, share-after-month, stressed card, busy leftover, report SIZE clock, 10-row hole, T1 leftover, Y6 size, fake all-out resid, days+size residual.
