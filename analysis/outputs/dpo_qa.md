# Q4/Q5 DPO leftover after days / DSO / issued_lag1

Generated `2026-09-19T04:46:40+02:00` by agent `a19d4e07`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_dpo`. Night Y7 quote stays **TURNOVER 0.720 / B_shallow 0.712**. TURNDPO 0.723 is **not** a KEEP. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. Delay/overdue locked. Do not grow TURNOVER or change 0.720.

`e_dpo_proxy` = AP open / this-period AP issued (months of billings outstanding). Same formula as DSO, payables side. Feature report: 51.2% cov, LOW_PERSIST acf 0.24, cluster rep; means unusable. Winsorise-at-24 is model-layer only.

## Headline

DPO vs DSO ρ=0.456 (not a twin). Y7 leftover after DSO 0.471 / issued_lag1 0.443 / both 0.463 — CLOSE. Y3 leftover after days 0.653 vs days 0.711 — CLOSE / DROP from the 44. Y3 leftover-after-days 0.653 is a |DPO|>24 tail (drop-tail leftover dies). Y5 AP single 0.445 leftover-paid 0.574 (not the Y). SIZE ρ=-0.025. Q6 CLOSE. DROP e_dpo_proxy from the 44.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_dpo`. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Not this clock. |
| 3 | Who is turning? | Q6 CLOSE — exists early (unlike delay; early6 Y7 finite 93.7%) but Y7 lag1 short 0.450 dies — stock/flow is not a lead. |
| 4 | Dip vs fall? | Y7 leftover CLOSE after DSO 0.471 / issued_lag1 0.443. |
| 5 | Why did it change? | Y3 leftover after days CLOSE / DROP from the 44 (0.653 vs days 0.711). Y5 never E as X. |
| 6 | Months earlier? | lag1 0.451 lag3 0.466 short lag1 0.450. Early6 finite 93.7%. |


## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| DPO as Y7 leftover after DSO + issued_lag1 | **CLOSE** | after DSO 0.471 / issued_lag1 0.443; twin=False; SIZE=False |
| DPO as Y7 add-on / grow TURNOVER / TURNDPO | **CLOSE** | TURNDPO 0.723 did not lift fold 4; night quote stays 0.720 / 0.712 |
| DPO as Y3 X / the 44 | **CLOSE / DROP from the 44** | leftover after days 0.653 vs days 0.711 size 0.617 |
| DPO as Y5 X | **FORBIDDEN (Y5 never E)** | single vs Y5 AP 0.445; leftover after delay_paid 0.574; not the Y |
| DPO as a health Y | **PARK** | do not invent `y_dpo` |
| Q6 DPO lag1/lag3 on short books | **CLOSE** | exists early (unlike delay; early6 Y7 finite 93.7%) but Y7 lag1 short 0.450 dies — stock/flow is not a lead |
| winsorise-at-24 in the store | **CLOSE** | model-layer only; clip leftover change=yes |
| e_dpo_proxy on the keep-list 44 | **DROP from the 44** | Y3 CLOSE / DROP from the 44; Y7 CLOSE |


## 1. Coverage / nulls / tails (dark = NaN not 0)

Train DPO nn=10,829 cov=51.2% (feature report 51.2%). Dark never-ERP 470 (want 470): nn=0 zero=0 (CONFIRM NaN not 0). Ever-ERP 744. Early6 finite 53.2% (delay was 0.0% — DPO is a stock/flow, may exist earlier). Tails p50=1.78 p99=802.4 max=114831.0 |DPO|>24 9.0% of finite (973). acf1=0.244 (CONFIRM LOW_PERSIST).

| col | nn | cov | early6 nn | after nn | dark nn / 0 | ERP nn | p50 | p99 | max | |DPO|>24 | acf1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e_dpo_proxy | 10,829 | 51.2% | 53.2% | 50.8% | 0 / 0 | 79.9% | 1.78 | 802.4 | 114831.0 | 9.0% | 0.244 |


Train CM 21,157 / companies train. Y7 labeled 7,464 pos 2,149. Y3 5,648 pos 402. Y5 AP 4,905 pos 418.

Plot: `dpo_vs_dso.png`.

## 2. Store vs raw SQL

Store vs raw SQL DPO ρ=1.000 max|Δ|=0.00000000 n_off=0 (CONFIRM SAME).

| pair | ρ | n | max|Δ| | n_off | only store | only SQL | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e_dpo_proxy vs sql_dpo | 1.000 | 10,829 | 0.00000000 | 0 | 0 | 0 | SAME |
| e_ap_open vs sql_ap_open | 1.000 | 13,554 | 0.00000000 | 0 | 0 | 0 | SAME |
| e_ap_issued vs sql_ap_issued | 1.000 | 13,554 | 0.00000000 | 0 | 1 | 0 | CLOSE |


## 3. Spearman twins (|ρ|≥0.80)

DPO vs DSO ρ=0.456 (not a twin). vs delay_paid 0.225 vs ap_overdue 0.500 vs ap_overdue_30 0.632 (not the Y). vs ap_issued -0.109 vs size -0.025 (not SIZE). TWIN |ρ|≥0.80: none.

| pair | ρ | n | twin? |
| --- | --- | --- | --- |
| e_dpo_proxy vs e_dso_proxy | 0.456 | 8,236 |  |
| e_dpo_proxy vs e_delay_paid | 0.225 | 8,060 |  |
| e_dpo_proxy vs e_ap_overdue | 0.500 | 10,654 |  |
| e_dpo_proxy vs e_ap_overdue_30 | 0.632 | 10,654 |  |
| e_dpo_proxy vs e_ap_issued | -0.109 | 10,829 |  |
| e_dpo_proxy vs e_pending_amt_share | 0.412 | 10,829 |  |
| e_dpo_proxy vs log1p(a_in3) | -0.025 | 9,801 |  |
| e_dpo_proxy vs c_n_days_with_tx | -0.027 | 10,829 |  |
| e_dpo_proxy vs f_ds_r | 0.022 | 9,801 |  |
| e_dpo_proxy vs e_delay_coll | 0.147 | 6,453 |  |
| e_dpo_proxy vs e_ar_issued | -0.002 | 10,829 |  |
| e_dpo_proxy vs c_zero_in_month | 0.002 | 10,829 |  |
| e_dpo_proxy vs c_zero_in_share_6 | -0.011 | 10,829 |  |


## 4. Single-feature train group-fold AUROC

Y3 DPO 0.627 vs days 0.711 (CONFIRM 0.711) vs size 0.617. Y7 DPO 0.450 vs DSO 0.431 vs issued_lag1 0.630 (night 0.630). Y5 AP DPO 0.445 vs delay_paid 0.591 — Y5 never E as X; this is leak/label only.

Sign from the train side of each fold. Seed 20260918. Night quotes: issued_lag1 0.630 (replica 0.630); days 0.711 (replica 0.711); size 0.617 (replica 0.617). Y5 never E as X — DPO vs Y5 is leak/label only.

| y | feature | n | n_pos | CV | sd | sign | folds | present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | e_dpo_proxy | 3,015 | 174 | 0.627 | 0.112 | 1 | 0.797 0.510 0.605 0.669 0.554 | 53.4% |
| y3_recover_cash_6m | e_dpo_clip24 | 3,015 | 174 | 0.627 | 0.111 | 1 | 0.794 0.512 0.607 0.670 0.552 | 53.4% |
| y3_recover_cash_6m | e_dso_proxy | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 | 42.8% |
| y3_recover_cash_6m | e_delay_paid | 2,294 | 150 | 0.521 | 0.098 | 1 | 0.419 0.441 0.543 0.665 0.536 | 40.6% |
| y3_recover_cash_6m | e_ap_overdue_30 | 3,336 | 223 | 0.620 | 0.142 | 1 | 0.787 0.548 0.491 0.759 0.515 | 59.1% |
| y3_recover_cash_6m | e_ap_issued | 3,618 | 264 | 0.675 | 0.117 | -1 | 0.811 0.656 0.492 0.712 0.704 | 64.1% |
| y3_recover_cash_6m | e_ar_issued_lag1 | 3,618 | 264 | 0.671 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 | 64.1% |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 | 100.0% |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 | 97.9% |
| y5_ap_od30_ownp80 | e_dpo_proxy | 4,814 | 415 | 0.445 | 0.072 | -1 | 0.425 0.535 0.359 0.502 0.406 | 98.1% |
| y5_ap_od30_ownp80 | e_dpo_clip24 | 4,814 | 415 | 0.446 | 0.072 | -1 | 0.426 0.535 0.360 0.502 0.405 | 98.1% |
| y5_ap_od30_ownp80 | e_dso_proxy | 3,946 | 351 | 0.470 | 0.082 | -1 | 0.444 0.549 0.566 0.400 0.392 | 80.4% |
| y5_ap_od30_ownp80 | e_delay_paid | 4,493 | 379 | 0.591 | 0.076 | -1 | 0.606 0.581 0.592 0.695 0.482 | 91.6% |
| y5_ap_od30_ownp80 | e_ap_overdue_30 | 4,890 | 418 | 0.596 | 0.105 | -1 | 0.645 0.652 0.679 0.586 0.418 | 99.7% |
| y5_ap_od30_ownp80 | e_ap_issued | 4,905 | 418 | 0.585 | 0.074 | 1 | 0.701 0.511 0.584 0.533 0.596 | 100.0% |
| y5_ap_od30_ownp80 | e_ar_issued_lag1 | 4,905 | 418 | 0.538 | 0.063 | 1 | 0.637 0.508 0.506 0.480 0.561 | 100.0% |
| y5_ap_od30_ownp80 | c_n_days_with_tx | 4,905 | 418 | 0.540 | 0.083 | 1 | 0.641 0.449 0.473 0.530 0.607 | 100.0% |
| y5_ap_od30_ownp80 | log1p_a_in3 | 4,905 | 418 | 0.556 | 0.084 | 1 | 0.674 0.539 0.466 0.495 0.606 | 100.0% |
| y7_top1_lost | e_dpo_proxy | 7,038 | 1,929 | 0.450 | 0.051 | -1 | 0.497 0.403 0.510 0.438 0.403 | 94.3% |
| y7_top1_lost | e_dpo_clip24 | 7,038 | 1,929 | 0.451 | 0.051 | -1 | 0.498 0.403 0.509 0.439 0.403 | 94.3% |
| y7_top1_lost | e_dso_proxy | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 | 89.1% |
| y7_top1_lost | e_delay_paid | 5,591 | 1,522 | 0.444 | 0.096 | -1 | 0.518 0.438 0.562 0.342 0.358 | 74.9% |
| y7_top1_lost | e_ap_overdue_30 | 7,278 | 2,086 | 0.416 | 0.069 | -1 | 0.489 0.370 0.495 0.370 0.356 | 97.5% |
| y7_top1_lost | e_ap_issued | 7,464 | 2,149 | 0.570 | 0.058 | -1 | 0.562 0.483 0.572 0.589 0.644 | 100.0% |
| y7_top1_lost | e_ar_issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 | 97.2% |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 | 100.0% |
| y7_top1_lost | log1p_a_in3 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 | 93.8% |


## 5. Honest leftover

Y3 leftover after days 0.653 (lives). Y7 leftover after DSO 0.471 / issued_lag1 0.443 / both 0.463 (dies). Y5 leftover after delay_paid 0.574 / od30 0.583 (diagnostic only; Y5 never E as X). Clip24 leftover Y3-days 0.569 Y7-DSO 0.462 Y7-iss 0.453.

Leftover <0.55 dies. Y5 leftover is diagnostic only.

| y | stem | residual | n | n_pos | CV | R² | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_dpo_proxy | after DSO | 6,389 | 1,536 | 0.471 | 0.003 | 0.529 0.463 0.534 0.449 0.381 |
| y7_top1_lost | e_dpo_proxy | after issued_lag1 | 6,865 | 1,870 | 0.443 | 0.000 | 0.504 0.406 0.478 0.433 0.395 |
| y7_top1_lost | e_dpo_proxy | after DSO+issued_lag1 | 6,238 | 1,490 | 0.463 | 0.005 | 0.532 0.467 0.499 0.442 0.373 |
| y7_top1_lost | e_dpo_proxy | after days | 7,038 | 1,929 | 0.453 | 0.002 | 0.436 0.448 0.469 0.465 0.445 |
| y7_top1_lost | e_dpo_proxy | after size | 6,650 | 1,798 | 0.453 | 0.000 | 0.468 0.517 0.462 0.480 0.338 |
| y7_top1_lost | e_dpo_proxy | after delay_paid | 5,430 | 1,428 | 0.465 | 0.001 | 0.554 0.440 0.571 0.402 0.360 |
| y7_top1_lost | e_dpo_clip24 | after DSO | 6,389 | 1,536 | 0.462 | 0.001 | 0.507 0.445 0.525 0.452 0.383 |
| y7_top1_lost | e_dpo_clip24 | after issued_lag1 | 6,865 | 1,870 | 0.453 | 0.000 | 0.501 0.407 0.513 0.443 0.399 |
| y7_top1_lost | e_dpo_clip24 | after DSO+issued_lag1 | 6,238 | 1,490 | 0.463 | 0.002 | 0.506 0.448 0.531 0.456 0.376 |
| y7_top1_lost | e_dpo_clip24 | after days | 7,038 | 1,929 | 0.446 | 0.016 | 0.482 0.401 0.489 0.461 0.398 |
| y7_top1_lost | e_dpo_clip24 | after size | 6,650 | 1,798 | 0.449 | 0.009 | 0.500 0.434 0.502 0.447 0.362 |
| y7_top1_lost | e_dpo_clip24 | after delay_paid | 5,430 | 1,428 | 0.530 | 0.036 | 0.515 0.590 0.538 0.518 0.487 |
| y3_recover_cash_6m | e_dpo_proxy | after days | 3,015 | 174 | 0.653 | 0.002 | 0.588 0.767 0.634 0.654 0.622 |
| y3_recover_cash_6m | e_dpo_proxy | after size | 2,959 | 168 | 0.459 | 0.000 | 0.449 0.343 0.565 0.444 0.491 |
| y3_recover_cash_6m | e_dpo_proxy | after DSO | 2,358 | 91 | 0.598 | 0.003 | 0.714 0.519 0.689 0.579 0.488 |
| y3_recover_cash_6m | e_dpo_proxy | after days+size | 2,959 | 168 | 0.627 | 0.002 | 0.565 0.744 0.579 0.623 0.627 |
| y3_recover_cash_6m | e_dpo_clip24 | after days | 3,015 | 174 | 0.569 | 0.016 | 0.714 0.438 0.550 0.632 0.509 |
| y3_recover_cash_6m | e_dpo_clip24 | after size | 2,959 | 168 | 0.609 | 0.009 | 0.797 0.493 0.581 0.637 0.539 |
| y3_recover_cash_6m | e_dpo_clip24 | after DSO | 2,358 | 91 | 0.599 | 0.001 | 0.711 0.514 0.689 0.581 0.501 |
| y3_recover_cash_6m | e_dpo_clip24 | after days+size | 2,959 | 168 | 0.563 | 0.023 | 0.706 0.439 0.552 0.613 0.505 |
| y5_ap_od30_ownp80 | e_dpo_proxy | after delay_paid | 4,441 | 378 | 0.574 | 0.001 | 0.586 0.554 0.556 0.678 0.497 |
| y5_ap_od30_ownp80 | e_dpo_proxy | after ap_overdue_30 | 4,805 | 415 | 0.583 | 0.003 | 0.638 0.623 0.670 0.570 0.416 |
| y5_ap_od30_ownp80 | e_dpo_proxy | after delay_paid+od30 | 4,432 | 378 | 0.598 | 0.004 | 0.649 0.628 0.679 0.615 0.419 |
| y5_ap_od30_ownp80 | e_dpo_proxy | after size | 4,814 | 415 | 0.440 | 0.000 | 0.410 0.489 0.413 0.485 0.404 |
| y5_ap_od30_ownp80 | e_dpo_clip24 | after delay_paid | 4,441 | 378 | 0.508 | 0.036 | 0.491 0.511 0.392 0.569 0.578 |
| y5_ap_od30_ownp80 | e_dpo_clip24 | after ap_overdue_30 | 4,805 | 415 | 0.524 | 0.345 | 0.529 0.554 0.513 0.535 0.490 |
| y5_ap_od30_ownp80 | e_dpo_clip24 | after delay_paid+od30 | 4,432 | 378 | 0.543 | 0.339 | 0.544 0.571 0.529 0.576 0.492 |
| y5_ap_od30_ownp80 | e_dpo_clip24 | after size | 4,814 | 415 | 0.469 | 0.009 | 0.554 0.534 0.348 0.511 0.400 |


## 6. Twin screen

TWIN |ρ|≥0.80 vs DSO / delay_paid / ap_overdue / ap_issued: none. No twin drop.

## 7. Winsorise-at-24 (in-memory only)

Clip24 single Y3 0.627 Y7 0.451. Clip leftover Y3-days 0.569 (raw leftover 0.653) Y7-DSO 0.462 (raw 0.471) Y7-iss 0.453. Clip moves leftover ≥0.02 — still do not write the clip to parquet.

## 8. SIZE terciles and invoice-book-only (drop 470)

Book-only (drop 470) Y3 0.627 Y7 0.450. SIZE T1 Y3 0.701 Y7 0.440 T3 Y3 — Y7 0.428.

| y | feature | n | n_pos | CV | sd | sign | folds | present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | DPO all | 3,015 | 174 | 0.627 | 0.112 | 1 | 0.797 0.510 0.605 0.669 0.554 | 53.4% |
| y3_recover_cash_6m | DPO leftover-days all | 3,015 | 174 | 0.653 | 0.068 | -1 | 0.588 0.767 0.634 0.654 0.622 |  |
| y7_top1_lost | DPO all | 7,038 | 1,929 | 0.450 | 0.051 | -1 | 0.497 0.403 0.510 0.438 0.403 | 94.3% |
| y7_top1_lost | DPO leftover-DSO all | 6,389 | 1,536 | 0.471 | 0.063 | -1 | 0.529 0.463 0.534 0.449 0.381 |  |
| y3_recover_cash_6m | DPO book_only | 3,015 | 174 | 0.627 | 0.112 | 1 | 0.797 0.510 0.605 0.669 0.554 | 83.3% |
| y3_recover_cash_6m | DPO leftover-days book_only | 3,015 | 174 | 0.653 | 0.068 | -1 | 0.588 0.767 0.634 0.654 0.622 |  |
| y7_top1_lost | DPO book_only | 7,038 | 1,929 | 0.450 | 0.051 | -1 | 0.497 0.403 0.510 0.438 0.403 | 94.3% |
| y7_top1_lost | DPO leftover-DSO book_only | 6,389 | 1,536 | 0.471 | 0.063 | -1 | 0.529 0.463 0.534 0.449 0.381 |  |
| y3_recover_cash_6m | DPO T1 | 564 | 70 | 0.701 | 0.133 | 1 | 0.905 0.641 0.671 0.741 0.548 | 48.8% |
| y3_recover_cash_6m | DPO leftover-days T1 | 564 | 70 | 0.525 | 0.113 | 1 | 0.537 0.336 0.615 0.611 0.525 |  |
| y7_top1_lost | DPO T1 | 1,659 | 503 | 0.440 | 0.062 | -1 | 0.484 0.497 0.424 0.342 0.453 | 92.6% |
| y7_top1_lost | DPO leftover-DSO T1 | 1,428 | 362 | 0.551 | 0.116 | -1 | 0.621 0.611 0.607 0.346 0.571 |  |
| y3_recover_cash_6m | DPO T2 | 1,122 | 49 | LOW_POWER | — | — | — | 56.6% |
| y7_top1_lost | DPO T2 | 2,593 | 721 | 0.462 | 0.037 | -1 | 0.443 0.498 0.490 0.472 0.408 | 95.4% |
| y3_recover_cash_6m | DPO T3 | 1,273 | 49 | LOW_POWER | — | — | — | 53.3% |
| y3_recover_cash_6m | DPO leftover-days T3 | 1,273 | 49 | LOW_POWER | — | — | — |  |
| y7_top1_lost | DPO T3 | 2,398 | 574 | 0.428 | 0.131 | -1 | 0.558 0.237 0.495 0.499 0.352 | 96.2% |
| y7_top1_lost | DPO leftover-DSO T3 | 2,223 | 472 | 0.431 | 0.133 | -1 | 0.545 0.251 0.520 0.513 0.327 |  |


## 9. Q6 — lag1/lag3; empty-on-short

Q6 Y7 DPO now short 0.451 lag1 0.450. Early6 calendar DPO finite share 93.7% (exists earlier than delay). CLOSE as Q6 if lag dies on short / empty den.

| slice | col | n_nn | n_pos | present | CV |
| --- | --- | --- | --- | --- | --- |
| all | e_dpo_proxy | 7,038 | 1,929 | 94.3% | 0.450 |
| all | e_dpo_proxy_lag1 | 6,803 | 1,878 | 91.1% | 0.451 |
| all | e_dpo_proxy_lag3 | 5,967 | 1,648 | 79.9% | 0.466 |
| all | e_dso_proxy | 6,651 | 1,623 | 89.1% | 0.431 |
| all | e_ar_issued_lag1 | 7,253 | 2,072 | 97.2% | 0.630 |
| all | e_delay_paid | 5,591 | 1,522 | 74.9% | 0.444 |
| short_<12_sofar | e_dpo_proxy | 4,110 | 1,187 | 93.0% | 0.451 |
| short_<12_sofar | e_dpo_proxy_lag1 | 3,879 | 1,129 | 87.7% | 0.450 |
| short_<12_sofar | e_dpo_proxy_lag3 | 3,080 | 889 | 69.7% | 0.507 |
| short_<12_sofar | e_dso_proxy | 3,925 | 1,015 | 88.8% | 0.428 |
| short_<12_sofar | e_ar_issued_lag1 | 4,210 | 1,260 | 95.2% | 0.626 |
| short_<12_sofar | e_delay_paid | 2,774 | 770 | 62.7% | 0.440 |
| long_>=18_sofar | e_dpo_proxy | 900 | 199 | 98.3% | 0.591 |
| long_>=18_sofar | e_dpo_proxy_lag1 | 903 | 202 | 98.6% | 0.583 |
| long_>=18_sofar | e_dpo_proxy_lag3 | 891 | 200 | 97.3% | 0.557 |
| long_>=18_sofar | e_dso_proxy | 828 | 157 | 90.4% | 0.607 |
| long_>=18_sofar | e_ar_issued_lag1 | 916 | 209 | 100.0% | 0.646 |
| long_>=18_sofar | e_delay_paid | 864 | 201 | 94.3% | 0.548 |
| short_<12_company | e_dpo_proxy | 711 | 150 | 91.9% | 0.625 |
| short_<12_company | e_dpo_proxy_lag1 | 591 | 118 | 76.4% | 0.674 |
| short_<12_company | e_dpo_proxy_lag3 | 329 | 57 | 42.5% | 0.652 |
| short_<12_company | e_dso_proxy | 704 | 127 | 91.0% | 0.634 |
| short_<12_company | e_ar_issued_lag1 | 666 | 140 | 86.0% | 0.722 |
| short_<12_company | e_delay_paid | 572 | 106 | 73.9% | 0.596 |
| long_>=18_company | e_dpo_proxy | 5,920 | 1,667 | 95.0% | 0.452 |
| long_>=18_company | e_dpo_proxy_lag1 | 5,837 | 1,656 | 93.7% | 0.452 |
| long_>=18_company | e_dpo_proxy_lag3 | 5,348 | 1,511 | 85.8% | 0.465 |
| long_>=18_company | e_dso_proxy | 5,539 | 1,408 | 88.9% | 0.385 |
| long_>=18_company | e_ar_issued_lag1 | 6,154 | 1,813 | 98.8% | 0.616 |
| long_>=18_company | e_delay_paid | 4,674 | 1,326 | 75.0% | 0.443 |
| early6_calendar | e_dpo_proxy | 1,003 | 350 | 93.7% | 0.400 |
| early6_calendar | e_dpo_proxy_lag1 | 929 | 315 | 86.7% | 0.377 |
| early6_calendar | e_dpo_proxy_lag3 | 605 | 200 | 56.5% | 0.368 |
| early6_calendar | e_dso_proxy | 935 | 299 | 87.3% | 0.536 |
| early6_calendar | e_ar_issued_lag1 | 1,002 | 357 | 93.6% | 0.588 |
| early6_calendar | e_delay_paid | 0 | 0 | 0.0% | LOW_POWER |
| after_month7 | e_dpo_proxy | 6,035 | 1,579 | 94.4% | 0.445 |
| after_month7 | e_dpo_proxy_lag1 | 5,874 | 1,563 | 91.9% | 0.441 |
| after_month7 | e_dpo_proxy_lag3 | 5,362 | 1,448 | 83.9% | 0.454 |
| after_month7 | e_dso_proxy | 5,716 | 1,324 | 89.4% | 0.380 |
| after_month7 | e_ar_issued_lag1 | 6,251 | 1,715 | 97.8% | 0.643 |
| after_month7 | e_delay_paid | 5,591 | 1,522 | 87.5% | 0.444 |


## 10. ICC / company-demean

DPO ICC=0.807 k=741 (DSO ICC=0.628). Y7 raw 0.450 demean 0.482 company-mean 0.417. acf1=0.244 (CONFIRM LOW_PERSIST month shock). not BETWEEN (≥0.85) — demean / month shock carries.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_dpo_proxy raw | 7,038 | 1,929 | 0.450 | 0.051 | -1 | 0.497 0.403 0.510 0.438 0.403 |
| y7_top1_lost | e_dpo_proxy demean | 7,038 | 1,929 | 0.482 | 0.044 | 1 | 0.544 0.463 0.510 0.442 0.450 |
| y7_top1_lost | e_dpo_proxy company-mean | 7,455 | 2,146 | 0.417 | 0.050 | 1 | 0.449 0.355 0.485 0.397 0.401 |
| y3_recover_cash_6m | e_dpo_proxy raw | 3,015 | 174 | 0.627 | 0.112 | 1 | 0.797 0.510 0.605 0.669 0.554 |
| y3_recover_cash_6m | e_dpo_proxy demean | 3,015 | 174 | 0.478 | 0.026 | -1 | 0.521 0.474 0.468 0.450 0.476 |
| y3_recover_cash_6m | e_dpo_proxy company-mean | 3,618 | 264 | 0.551 | 0.102 | 1 | 0.518 0.456 0.497 0.718 0.567 |


## 11. vs Y6 / zero-in (do not revive Y6)

DPO vs zero-in month ρ=0.002 vs y6_zero_in_3 ρ=0.006. Y6 zero-in DPO 0.415 vs days 0.865. not an inverse-activity dummy. Do not revive Y6.

| vs | ρ | n | dummy? |
| --- | --- | --- | --- |
| c_zero_in_month | 0.002 | 10,829 |  |
| c_zero_in_share_6 | -0.011 | 10,829 |  |
| y6_zero_in_3 | 0.006 | 8,948 |  |
| y6_missed_payroll | 0.085 | 3,609 |  |
| c_n_days_with_tx | -0.027 | 10,829 |  |
| log1p(a_in3) | -0.025 | 9,801 |  |


| y | feature | n | n_pos | CV | sd | sign | folds | present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y6_zero_in_3 | e_dpo_proxy | 8,948 | 489 | 0.415 | 0.026 | 1 | 0.438 0.413 0.384 0.396 0.446 | 52.4% |
| y6_zero_in_3 | c_zero_in_month | 17,083 | 1,490 | 0.694 | 0.039 | 1 | 0.666 0.757 0.704 0.679 0.662 | 100.0% |
| y6_zero_in_3 | c_n_days_with_tx | 17,083 | 1,490 | 0.865 | 0.017 | -1 | 0.842 0.884 0.852 0.871 0.874 | 100.0% |
| y6_zero_in_3 | log1p_a_in3 | 14,698 | 1,300 | 0.854 | 0.021 | -1 | 0.836 0.848 0.850 0.844 0.890 | 86.0% |
| y6_missed_payroll | e_dpo_proxy | 3,609 | 181 | 0.638 | 0.072 | 1 | 0.723 0.700 0.624 0.591 0.552 | 60.1% |
| y6_missed_payroll | c_zero_in_month | 6,004 | 367 | 0.535 | 0.024 | 1 | 0.507 0.563 0.555 0.518 0.532 | 100.0% |
| y6_missed_payroll | c_n_days_with_tx | 6,004 | 367 | 0.599 | 0.060 | -1 | 0.541 0.556 0.663 0.571 0.665 | 100.0% |
| y6_missed_payroll | log1p_a_in3 | 6,004 | 367 | 0.562 | 0.046 | -1 | 0.512 0.593 0.573 0.615 0.515 | 100.0% |


## Extra — quintiles of DPO vs Y3 / Y5 / Y7

DPO quintiles vs Y3/Y5/Y7. Q5 tail on Y3 (high-DPO / thin issued).

| y | DPO q | n | n_pos | rate | DPO p50 | issued p50 | share>24 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | Q1 | 603 | 30 | 5.0% | 0.33 | 42334 | 0.0% |
| y3_recover_cash_6m | Q2 | 603 | 32 | 5.3% | 1.00 | 49902 | 0.0% |
| y3_recover_cash_6m | Q3 | 603 | 19 | 3.2% | 1.70 | 91104 | 0.0% |
| y3_recover_cash_6m | Q4 | 603 | 24 | 4.0% | 3.51 | 84881 | 0.0% |
| y3_recover_cash_6m | Q5 | 603 | 69 | 11.4% | 14.08 | 21979 | 36.8% |
| y5_ap_od30_ownp80 | Q1 | 963 | 68 | 7.1% | 0.48 | 64412 | 0.0% |
| y5_ap_od30_ownp80 | Q2 | 963 | 93 | 9.7% | 1.18 | 65673 | 0.0% |
| y5_ap_od30_ownp80 | Q3 | 962 | 107 | 11.1% | 2.22 | 116711 | 0.0% |
| y5_ap_od30_ownp80 | Q4 | 963 | 94 | 9.8% | 4.84 | 81285 | 0.0% |
| y5_ap_od30_ownp80 | Q5 | 963 | 53 | 5.5% | 21.65 | 13921 | 45.7% |
| y7_top1_lost | Q1 | 1408 | 424 | 30.1% | 0.39 | 51252 | 0.0% |
| y7_top1_lost | Q2 | 1407 | 417 | 29.6% | 1.04 | 61293 | 0.0% |
| y7_top1_lost | Q3 | 1408 | 321 | 22.8% | 1.82 | 87245 | 0.0% |
| y7_top1_lost | Q4 | 1407 | 358 | 25.4% | 3.81 | 87547 | 0.0% |
| y7_top1_lost | Q5 | 1408 | 409 | 29.0% | 17.50 | 16657 | 39.8% |


## Extra — fold 4 / short-DSO quintile

Fold 4 Y7 DPO 0.403 vs DSO 0.342 vs issued_lag1 0.647 vs TURNOVER 0.680. Short-DSO Q1: DPO 0.563 vs DSO 0.455. DPO lifts the short-DSO fifth.

| feature | CV | fold4 | folds |
| --- | --- | --- | --- |
| e_dpo_proxy | 0.450 | 0.403 | 0.497 0.403 0.510 0.438 0.403 |
| e_dpo_clip24 | 0.451 | 0.403 | 0.498 0.403 0.509 0.439 0.403 |
| e_dso_proxy | 0.431 | 0.342 | 0.370 0.600 0.420 0.424 0.342 |
| e_ar_issued_lag1 | 0.630 | 0.647 | 0.643 0.662 0.590 0.605 0.647 |
| e_delay_paid | 0.444 | 0.358 | 0.518 0.438 0.562 0.342 0.358 |
| c_n_days_with_tx | 0.458 | 0.439 | 0.457 0.483 0.478 0.432 0.439 |


| DSO q | n | rate | DSO p50 | DPO nn | DPO CV | DSO CV | issued CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | 1331 | 25.5% | 0.06 | 97.1% | 0.563 | 0.455 | 0.604 |
| Q2 | 1330 | 25.0% | 1.00 | 95.6% | 0.464 | 0.449 | 0.647 |
| Q3 | 1330 | 22.1% | 1.74 | 96.8% | 0.557 | 0.582 | 0.448 |
| Q4 | 1330 | 19.8% | 3.74 | 95.8% | 0.539 | 0.546 | 0.521 |
| Q5 | 1330 | 29.6% | 15.45 | 95.0% | 0.531 | 0.644 | 0.644 |


## Extra — holdout coverage only (no AUROC)

Holdout 72 coverage only: 1,073 CM / 72 companies. Y7 pos=122 (quote 122). DPO cov 47.3%; early6 23.6%; dark nn=0. No AUROC claim.

| col | n_cm | nn | cov | early6 nn | dark nn | |DPO|>24 |
| --- | --- | --- | --- | --- | --- | --- |
| e_dpo_proxy | 1,073 | 507 | 47.3% | 23.6% | 0 | 11.8% |


## Extra — same-n leftover

Same-n (DPO+DSO+issued_lag1 finite): raw DPO 0.460 leftover 0.463 issued 0.583 DSO 0.392 R²=0.005. Leftover ≈ raw — residualizing does not create a new object.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | DPO same-n | 6,238 | 1,490 | 0.460 | 0.059 | -1 | 0.504 0.446 0.527 0.450 0.376 |
| y7_top1_lost | issued_lag1 same-n | 6,238 | 1,490 | 0.583 | 0.043 | -1 | 0.592 0.569 0.538 0.566 0.652 |
| y7_top1_lost | DSO same-n | 6,238 | 1,490 | 0.392 | 0.042 | 1 | 0.370 0.392 0.432 0.431 0.333 |
| y7_top1_lost | leftover both same-n | 6,238 | 1,490 | 0.463 | 0.060 | -1 | 0.532 0.467 0.499 0.442 0.373 |
| y7_top1_lost | clip24 same-n | 6,238 | 1,490 | 0.461 | 0.058 | -1 | 0.504 0.447 0.526 0.451 0.376 |


## Extra — tiny-issued blow-up

Tiny-issued p10=798: DPO p50 on tiny 4.15 share>24 31.9%. Y7 DPO on tiny 0.553 vs rest 0.450.

| slice | n | DPO nn | DPO p50 | DPO p99 | |DPO|>24 | issued p50 |
| --- | --- | --- | --- | --- | --- | --- |
| tiny_issued p10 | 1083 | 1083 | 4.15 | 30429.7 | 31.9% | 260 |
| rest issued>p10 | 9746 | 9746 | 1.71 | 235.5 | 6.4% | 61258 |
| issued=0 / NaN DPO | 10328 | 0 | — | — | — | 0 |


## Extra — is Y3 leftover a tail / thin-issued artifact?

Y3 leftover after days on drop>|24| 0.432 clip12 0.584 after log1p(issued) 0.597 days+issued 0.603. has_dpo 0.552 has_tiny 0.528 has>24 0.526. Leftover is a tail / thin-issued artifact.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover days (raw) | 3,015 | 174 | 0.653 | 0.068 | -1 | 0.588 0.767 0.634 0.654 0.622 |
| y3_recover_cash_6m | leftover after log1p(ap_issued) | 3,015 | 174 | 0.597 | 0.073 | -1 | 0.694 0.596 0.500 0.564 0.633 |
| y3_recover_cash_6m | leftover days+issued | 3,015 | 174 | 0.603 | 0.073 | -1 | 0.697 0.606 0.503 0.572 0.638 |
| y3_recover_cash_6m | leftover days drop |DPO|>24 | 2,793 | 137 | 0.432 | 0.114 | 1 | 0.271 0.532 0.402 0.551 0.406 |
| y3_recover_cash_6m | leftover days clip12 | 3,015 | 174 | 0.584 | 0.114 | 1 | 0.752 0.459 0.571 0.630 0.508 |
| y3_recover_cash_6m | leftover days clip24 | 3,015 | 174 | 0.569 | 0.107 | 1 | 0.714 0.438 0.550 0.632 0.509 |
| y3_recover_cash_6m | has_dpo flag | 5,648 | 402 | 0.552 | 0.040 | -1 | 0.535 0.528 0.514 0.613 0.572 |
| y3_recover_cash_6m | has_tiny_issued | 5,648 | 402 | 0.528 | 0.023 | 1 | 0.534 0.508 0.526 0.510 0.564 |
| y3_recover_cash_6m | has_|DPO|>24 | 5,648 | 402 | 0.526 | 0.030 | 1 | 0.531 0.485 0.509 0.549 0.556 |
| y3_recover_cash_6m | log1p(ap_issued) | 3,618 | 264 | 0.675 | 0.117 | -1 | 0.811 0.656 0.492 0.712 0.704 |
| y3_recover_cash_6m | DPO drop |x|>24 raw | 2,793 | 137 | 0.582 | 0.118 | 1 | 0.749 0.561 0.615 0.563 0.421 |
| y3_recover_cash_6m | DPO clip12 raw | 3,015 | 174 | 0.625 | 0.109 | 1 | 0.791 0.515 0.609 0.664 0.548 |
| y7_top1_lost | leftover days drop |DPO|>24 | 6,477 | 1,721 | 0.459 | 0.065 | -1 | 0.486 0.419 0.542 0.475 0.372 |
| y7_top1_lost | leftover after log1p(ap_issued) | 7,038 | 1,929 | 0.463 | 0.075 | -1 | 0.479 0.405 0.528 0.538 0.367 |
| y3_recover_cash_6m | leftover days T1 | 564 | 70 | 0.525 | 0.113 | 1 | 0.537 0.336 0.615 0.611 0.525 |
| y3_recover_cash_6m | leftover days T2 | 1,122 | 49 | LOW_POWER | — | — | — |
| y3_recover_cash_6m | DPO exclude Q5-ish |x|>24 | 2,793 | 137 | 0.582 | 0.118 | 1 | 0.749 0.561 0.615 0.563 0.421 |


## Extra — Y3 Q6 lag / early leftover

Y3 Q6 DPO lag1 0.610 short 0.612. Early6 leftover-days — vs after-month7 0.663.

| slice | col | n_nn | n_pos | present | CV |
| --- | --- | --- | --- | --- | --- |
| all | e_dpo_proxy | 3,015 | 174 | 53.4% | 0.627 |
| all | e_dpo_proxy_lag1 | 2,991 | 177 | 53.0% | 0.610 |
| all | e_dpo_proxy_lag3 | 2,654 | 150 | 47.0% | 0.617 |
| all | c_n_days_with_tx | 5,648 | 402 | 100.0% | 0.711 |
| all | leftover days | 3,015 | 174 | 53.4% | 0.653 |
| short_<12_sofar | e_dpo_proxy | 1,934 | 109 | 51.9% | 0.625 |
| short_<12_sofar | e_dpo_proxy_lag1 | 1,911 | 109 | 51.3% | 0.612 |
| short_<12_sofar | e_dpo_proxy_lag3 | 1,576 | 81 | 42.3% | 0.591 |
| short_<12_sofar | c_n_days_with_tx | 3,723 | 252 | 100.0% | 0.696 |
| short_<12_sofar | leftover days | 1,934 | 109 | 51.9% | 0.671 |
| early6_calendar | e_dpo_proxy | 564 | 25 | 55.9% | LOW_POWER |
| early6_calendar | e_dpo_proxy_lag1 | 555 | 24 | 55.0% | LOW_POWER |
| early6_calendar | e_dpo_proxy_lag3 | 360 | 15 | 35.7% | LOW_POWER |
| early6_calendar | c_n_days_with_tx | 1,009 | 45 | 100.0% | LOW_POWER |
| early6_calendar | leftover days | 564 | 25 | 55.9% | LOW_POWER |
| after_month7 | e_dpo_proxy | 2,451 | 149 | 52.8% | 0.618 |
| after_month7 | e_dpo_proxy_lag1 | 2,436 | 153 | 52.5% | 0.627 |
| after_month7 | e_dpo_proxy_lag3 | 2,294 | 135 | 49.5% | 0.638 |
| after_month7 | c_n_days_with_tx | 4,639 | 357 | 100.0% | 0.722 |
| after_month7 | leftover days | 2,451 | 149 | 52.8% | 0.663 |
| long_>=18_sofar | e_dpo_proxy | 117 | 7 | 54.9% | LOW_POWER |
| long_>=18_sofar | e_dpo_proxy_lag1 | 116 | 7 | 54.5% | LOW_POWER |
| long_>=18_sofar | e_dpo_proxy_lag3 | 115 | 6 | 54.0% | LOW_POWER |
| long_>=18_sofar | c_n_days_with_tx | 213 | 16 | 100.0% | LOW_POWER |
| long_>=18_sofar | leftover days | 117 | 7 | 54.9% | LOW_POWER |


## Extra — Y3 same-n leftover vs days / issued

Y3 same-n raw 0.627 leftover-days 0.653 days 0.732 issued 0.648 R²=0.002. Demean leftover 0.656 mean leftover 0.626.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | DPO same-n | 3,015 | 174 | 0.627 | 0.112 | 1 | 0.797 0.510 0.605 0.669 0.554 |
| y3_recover_cash_6m | days same-n | 3,015 | 174 | 0.732 | 0.054 | -1 | 0.738 0.747 0.639 0.780 0.755 |
| y3_recover_cash_6m | ap_issued same-n | 3,015 | 174 | 0.648 | 0.122 | -1 | 0.805 0.578 0.495 0.635 0.727 |
| y3_recover_cash_6m | leftover days same-n | 3,015 | 174 | 0.653 | 0.068 | -1 | 0.588 0.767 0.634 0.654 0.622 |
| y3_recover_cash_6m | clip24 same-n | 3,015 | 174 | 0.627 | 0.111 | 1 | 0.794 0.512 0.607 0.670 0.552 |
| y3_recover_cash_6m | size same-n | 2,959 | 168 | 0.631 | 0.016 | -1 | 0.636 0.638 0.648 0.626 0.605 |
| y3_recover_cash_6m | demean leftover days | 3,015 | 174 | 0.656 | 0.058 | -1 | 0.711 0.710 0.571 0.634 0.655 |
| y3_recover_cash_6m | company-mean leftover days | 3,618 | 264 | 0.626 | 0.088 | -1 | 0.642 0.758 0.629 0.519 0.581 |
| y3_recover_cash_6m | demean raw | 3,015 | 174 | 0.478 | 0.026 | -1 | 0.521 0.474 0.468 0.450 0.476 |
| y3_recover_cash_6m | company-mean raw | 3,618 | 264 | 0.551 | 0.102 | 1 | 0.518 0.456 0.497 0.718 0.567 |


## Extra — fold-4 problem groups

Fold-4 problem groups DPO p50 vs rest (table). Y7 DPO on problem groups 0.573 vs rest 0.538.

| slice | n_lab | n_pos | rate | DPO nn | DPO p50 | DSO p50 |
| --- | --- | --- | --- | --- | --- | --- |
| fold4 problem groups | 540 | 371 | 68.7% | 98.0% | 0.88 | 0.18 |
| rest labeled | 6924 | 1778 | 25.7% | 94.0% | 1.99 | 1.97 |


## Extra — ICC on clip24

ICC raw 0.807 clip24 0.908. acf1 clip24 0.300 (raw LOW_PERSIST 0.24). clip makes BETWEEN.

## Extra — Y3/Y7 lag1 leftover after days

Y3 lag1 leftover after days 0.665 drop>24 0.433 (now leftover 0.653). Q6 leftover is the same tail.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | lag1 raw | 2,991 | 177 | 0.610 | 0.089 | 1 | 0.683 0.510 0.602 0.716 0.539 |
| y3_recover_cash_6m | lag1 leftover days | 2,991 | 177 | 0.665 | 0.062 | -1 | 0.662 0.772 0.627 0.624 0.638 |
| y3_recover_cash_6m | lag1 leftover days drop>24 | 2,792 | 139 | 0.433 | 0.025 | 1 | 0.417 0.469 0.442 0.404 0.435 |
| y3_recover_cash_6m | now leftover days | 3,015 | 174 | 0.653 | 0.068 | -1 | 0.588 0.767 0.634 0.654 0.622 |
| y7_top1_lost | lag1 raw | 6,803 | 1,878 | 0.451 | 0.045 | -1 | 0.468 0.422 0.514 0.454 0.397 |
| y7_top1_lost | lag1 leftover days | 6,803 | 1,878 | 0.470 | 0.043 | 1 | 0.425 0.540 0.470 0.473 0.444 |
| y7_top1_lost | lag1 leftover days drop>24 | 6,297 | 1,683 | 0.463 | 0.068 | -1 | 0.458 0.438 0.542 0.509 0.367 |
| y7_top1_lost | now leftover days | 7,038 | 1,929 | 0.453 | 0.014 | 1 | 0.436 0.448 0.469 0.465 0.445 |


## Extra — Y5 leftover drop-tail (diagnostic)

Y5 leftover after delay_paid 0.574 drop>24 0.527 clip24 0.508 after od30 0.583 R²=0.001. Diagnostic leftover dies on drop-tail — not a Y5 X anyway.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y5_ap_od30_ownp80 | DPO raw | 4,814 | 415 | 0.445 | 0.072 | -1 | 0.425 0.535 0.359 0.502 0.406 |
| y5_ap_od30_ownp80 | leftover delay_paid | 4,441 | 378 | 0.574 | 0.066 | 1 | 0.586 0.554 0.556 0.678 0.497 |
| y5_ap_od30_ownp80 | leftover paid drop>24 | 4,071 | 362 | 0.527 | 0.074 | 1 | 0.535 0.546 0.400 0.559 0.595 |
| y5_ap_od30_ownp80 | leftover od30 | 4,805 | 415 | 0.583 | 0.101 | 1 | 0.638 0.623 0.670 0.570 0.416 |
| y5_ap_od30_ownp80 | clip24 leftover paid | 4,441 | 378 | 0.508 | 0.075 | 1 | 0.491 0.511 0.392 0.569 0.578 |


## Extra — chronic 12 (Y3 only)

Chronic 12 Y2 names: 12. Y3 DPO 0.627 → drop-12 0.627. Days 0.711 → 0.708. does not flip.

| slice | DPO | days |
| --- | --- | --- |
| Y3 all | 0.627 | 0.711 |
| Y3 drop-12 | 0.627 | 0.708 |


## Extra — Y3 fold-wise leftover

Y3 fold-wise leftover-days 0.653 folds 0.588 0.767 0.634 0.654 0.622 vs days 0.665 0.738 0.700 0.715 0.740.

| feature | CV | folds | fold4 |
| --- | --- | --- | --- |
| e_dpo_proxy | 0.627 | 0.797 0.510 0.605 0.669 0.554 | 0.554 |
| c_n_days_with_tx | 0.711 | 0.665 0.738 0.700 0.715 0.740 | 0.740 |
| leftover days | 0.653 | 0.588 0.767 0.634 0.654 0.622 | 0.622 |
| e_dpo_clip24 | 0.627 | 0.794 0.512 0.607 0.670 0.552 | 0.552 |


## Extra — SIZE T1 leftover vs drop-tail

T1 Y3 DPO 0.701 leftover-days 0.525 drop>24 0.706 vs days 0.615.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | T1 DPO | 564 | 70 | 0.701 | 0.133 | 1 | 0.905 0.641 0.671 0.741 0.548 |
| y3_recover_cash_6m | T1 leftover days | 564 | 70 | 0.525 | 0.113 | 1 | 0.537 0.336 0.615 0.611 0.525 |
| y3_recover_cash_6m | T1 leftover drop>24 | 510 | 56 | 0.706 | 0.154 | 1 | 0.923 0.724 0.719 0.676 0.490 |
| y3_recover_cash_6m | T1 days | 1,156 | 170 | 0.615 | 0.024 | -1 | 0.595 0.648 0.588 0.622 0.623 |


## Extra — fat issued (issued > p10) leftover

Fat issued>p10=798: Y3 DPO 0.621 leftover-days 0.662 vs days 0.718. Y7 leftover 0.465.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | fat issued DPO | 2,753 | 133 | 0.621 | 0.138 | 1 | 0.826 0.489 0.636 0.653 0.499 |
| y3_recover_cash_6m | fat leftover days | 2,753 | 133 | 0.662 | 0.065 | -1 | 0.599 0.753 0.597 0.678 0.685 |
| y3_recover_cash_6m | fat days | 2,753 | 133 | 0.718 | 0.074 | -1 | 0.706 0.733 0.602 0.804 0.744 |
| y7_top1_lost | fat issued DPO | 6,554 | 1,797 | 0.450 | 0.050 | -1 | 0.465 0.416 0.526 0.444 0.399 |
| y7_top1_lost | fat leftover days | 6,554 | 1,797 | 0.465 | 0.016 | -1 | 0.485 0.462 0.477 0.450 0.449 |
| y7_top1_lost | fat days | 6,554 | 1,797 | 0.464 | 0.027 | -1 | 0.498 0.471 0.476 0.432 0.442 |


## What failed / next (held for wave note)

- Y7 leftover dies DSO 0.471 / issued 0.443
- DROP e_dpo_proxy from the 44
- Q6 CLOSE — exists early (unlike delay; early6 Y7 finite 93.7%) but Y7 lag1 short 0.450 dies — stock/flow is not a lead
- do not grow TURNOVER; TURNDPO 0.723 is not KEEP
- Y5 never E as X
- Y3 leftover after days is a tail/thin-issued artifact (drop>24 0.432 clip12 0.584 after-issued 0.597)
- Y3 Q6 lag1 0.610 short 0.612
- Y3 lag1 leftover-days 0.665 drop>24 0.433
- Y5 leftover drop>24 0.527 — still never E
- clip24 ICC 0.908 BETWEEN — do not write clip to store
- T1 leftover drop>24 0.706 vs days 0.615
- fat-issued leftover Y3 0.662 vs days 0.718

Elapsed 7s. Cuts: coverage, SQL, twins, singles, leftover, SIZE/book, Q6, ICC, Y6, quintiles, fold 4, holdout, same-n, tiny-issued, tail leftover, Y3 Q6, Y3 same-n, fold4 groups, clip ICC, lag leftover, Y5 tail, chronic-12, Y3 folds, T1 tail, fat issued.
