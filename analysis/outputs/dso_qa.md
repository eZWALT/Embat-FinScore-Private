# Q4/Q5 unused leftover of `e_dso_proxy` on the 44

Generated `2026-09-19T05:16:11+02:00` by agent `e8b2c0d4`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_dso`. Night Y7 quote stays **TURNOVER 0.720 / B_shallow 0.712**. Even card **drops DSO**. Do **not** put DSO back on TURNOVER. Do not change 0.720. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. DPO just DROPPED from the 44. Delay leftover KEEP after DSO (0.581). Winsorise-at-24 is model-layer only.

`e_dso_proxy` = AR open / this-period AR issued (months of billings outstanding). Feature report: 40.6% cov, acf1 0.25, ICC 0.63; means unusable. DSO is already CLOSE as a TURNOVER stem. This card is the unused leftover after days (Y3 X) and after issued_lag1 (Y7), plus leftover after delay_coll.

## Headline

DSO vs DPO ρ=0.456 (not a twin). vs delay_coll 0.214 (not a twin). Y7 leftover after issued_lag1 0.452 — CLOSE. Y3 leftover after days 0.474 vs days 0.711 — CLOSE / DROP from the 44. Y3 leftover-after-days 0.474 is not only the |DSO|>24 tail (drop-tail 0.424). Y7 leftover after delay 0.561 (not a delay twin; leftover-after-delay is a |DSO|>24 tail). Short-DSO Q1 univariate 0.455 (B_shallow OOF 0.410 locked). Fold 4 DSO 0.342 vs issued 0.647. SIZE ρ=0.064. Q6 CLOSE. DROP e_dso_proxy from the 44. Do not put DSO back on TURNOVER 0.720.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_dso`. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Not this clock. |
| 3 | Who is turning? | Q6 CLOSE — exists early (unlike delay; early6 Y7 finite 87.3%) but Y7 lag1 short 0.422 dies — stock/flow is not a TURNOVER lead. |
| 4 | Dip vs fall? | Y7 leftover after issued_lag1 CLOSE (0.452). DSO is CLOSE as a TURNOVER stem — do not grow 0.720. Fold 4 issued 0.647 vs DSO 0.342. |
| 5 | Why did it change? | Y3 leftover after days CLOSE / DROP from the 44 (0.474 vs days 0.711). SHAP #1 is not a delay twin; leftover-after-delay is a |DSO|>24 tail (leftover after delay 0.561). |
| 6 | Months earlier? | lag1 0.401 lag3 0.396 short lag1 0.422. Early6 finite 87.3%. |


## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| DSO as Y7 leftover after issued_lag1 | **CLOSE** | after issued_lag1 0.452; twin=False; SIZE=False |
| DSO as Y7 TURNOVER stem / put DSO back | **CLOSE** | even card drops DSO; TURNOVER 0.720 / fold 4 0.680 is issued; do not change 0.720 |
| DSO as Y3 X / the 44 | **CLOSE / DROP from the 44** | leftover after days 0.474 vs days 0.711 size 0.617; drop-tail 0.424 |
| DSO leftover after delay_coll (SHAP #1?) | **not a delay twin; leftover-after-delay is a |DSO|>24 tail** | leftover after delay 0.561 same-n 0.561; delay leftover KEEP 0.581 |
| DSO as a health Y | **PARK** | do not invent `y_dso` |
| Q6 DSO lag1/lag3 on short books | **CLOSE** | exists early (unlike delay; early6 Y7 finite 87.3%) but Y7 lag1 short 0.422 dies — stock/flow is not a TURNOVER lead |
| winsorise-at-24 in the store | **CLOSE** | model-layer only; clip leftover change=yes |
| e_dso_proxy on the keep-list 44 | **DROP from the 44** | Y3 CLOSE / DROP from the 44; Y7 CLOSE |


## 1. Coverage / nulls / tails (dark = NaN not 0)

Train DSO nn=8,583 cov=40.6% (feature report 40.6%). Dark never-ERP 470 (want 470): nn=0 zero=0 (CONFIRM NaN not 0). Ever-ERP 744. Early6 finite 40.2% (delay was 0.0% — DSO is a stock/flow, may exist earlier like DPO). Tails p50=1.74 p99=747.3 max=2190260.7 |DSO|>24 7.2% of finite (616). |DSO|>24 7.2% vs locked DPO 9.0% (this-run DPO 9.0%); DSO p99=747.3 vs DPO p99 802 (this-run 802.4). acf1=0.255 (not LOW_PERSIST / borderline).

| col | nn | cov | early6 nn | after nn | dark nn / 0 | ERP nn | p50 | p99 | max | |DSO|>24 | acf1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e_dso_proxy | 8,583 | 40.6% | 40.2% | 40.6% | 0 / 0 | 63.3% | 1.74 | 747.3 | 2190260.7 | 7.2% | 0.255 |
| e_dpo_proxy (locked) | 10,829 | 51.2% | — | — | — | — | 1.78 | 802.4 | 114831.0 | 9.0% | 0.244 |


Train CM 21,157 / companies train. Y7 labeled 7,464 pos 2,149. Y3 5,648 pos 402.

Plot: `dso_leftover.png`.

## 2. Store vs raw SQL

Store vs raw SQL DSO ρ=1.000 max|Δ|=0.00000000 n_off=0 (CONFIRM SAME).

| pair | ρ | n | max|Δ| | n_off | only store | only SQL | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e_dso_proxy vs sql_dso | 1.000 | 8,583 | 0.00000000 | 0 | 0 | 0 | SAME |
| e_ar_open vs sql_ar_open | 1.000 | 13,554 | 0.00000000 | 0 | 0 | 0 | SAME |
| e_ar_issued vs sql_ar_issued | 1.000 | 13,554 | 0.00000000 | 0 | 1 | 0 | CLOSE |


## 3. Spearman twins (|ρ|≥0.80)

DSO vs DPO ρ=0.456 (not a twin — locked 0.456). vs delay_coll 0.214 (not a twin — delay locked 0.214). vs ar_overdue 0.229 vs ar_overdue_30 0.534 vs issued -0.048 vs issued_lag1 0.147 vs days 0.022 vs size 0.064 (not SIZE). TWIN |ρ|≥0.80: none.

| pair | ρ | n | twin? |
| --- | --- | --- | --- |
| e_dso_proxy vs e_dpo_proxy | 0.456 | 8,236 |  |
| e_dso_proxy vs e_delay_coll | 0.214 | 6,091 |  |
| e_dso_proxy vs e_ar_overdue | 0.229 | 8,029 |  |
| e_dso_proxy vs e_ar_overdue_30 | 0.534 | 8,029 |  |
| e_dso_proxy vs e_ar_issued | -0.048 | 8,583 |  |
| e_dso_proxy vs e_ar_issued_lag1 | 0.147 | 8,214 |  |
| e_dso_proxy vs e_ar_open | 0.600 | 8,583 |  |
| e_dso_proxy vs e_pending_amt_share | 0.492 | 8,583 |  |
| e_dso_proxy vs log1p(a_in3) | 0.064 | 7,813 |  |
| e_dso_proxy vs c_n_days_with_tx | 0.022 | 8,583 |  |
| e_dso_proxy vs e_delay_paid | 0.216 | 6,441 |  |
| e_dso_proxy vs e_ap_overdue | 0.288 | 8,374 |  |
| e_dso_proxy vs c_zero_in_month | 0.053 | 8,583 |  |
| e_dso_proxy vs f_ds_r | 0.008 | 7,813 |  |


## 4. Single-feature train group-fold AUROC

Y3 DSO 0.564 vs days 0.711 (CONFIRM 0.711) vs size 0.617 (CONFIRM 0.617). Y7 DSO 0.431 vs issued_lag1 0.630 (CONFIRM 0.630) vs delay_coll 0.574 vs DPO 0.450 vs size 0.469.

Sign from the train side of each fold. Seed 20260918. Night quotes: issued_lag1 0.630 (replica 0.630); days 0.711 (replica 0.711); size 0.617 (replica 0.617). Do not put DSO back on TURNOVER.

| y | feature | n | n_pos | CV | sd | sign | folds | present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | e_dso_proxy | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 | 42.8% |
| y3_recover_cash_6m | e_dso_clip24 | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.382 0.587 0.697 0.475 | 42.8% |
| y3_recover_cash_6m | e_dso_clip12 | 2,418 | 93 | 0.564 | 0.134 | 1 | 0.675 0.381 0.595 0.697 0.474 | 42.8% |
| y3_recover_cash_6m | e_dpo_proxy | 3,015 | 174 | 0.627 | 0.112 | 1 | 0.797 0.510 0.605 0.669 0.554 | 53.4% |
| y3_recover_cash_6m | e_delay_coll | 1,819 | 86 | 0.512 | 0.087 | 1 | 0.426 0.455 0.626 0.582 0.473 | 32.2% |
| y3_recover_cash_6m | e_ar_overdue | 2,655 | 143 | 0.616 | 0.108 | 1 | 0.776 0.591 0.558 0.661 0.493 | 47.0% |
| y3_recover_cash_6m | e_ar_issued | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 | 64.1% |
| y3_recover_cash_6m | e_ar_issued_lag1 | 3,618 | 264 | 0.671 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 | 64.1% |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 | 100.0% |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 | 97.9% |
| y7_top1_lost | e_dso_proxy | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 | 89.1% |
| y7_top1_lost | e_dso_clip24 | 6,651 | 1,623 | 0.391 | 0.035 | 1 | 0.371 0.400 0.421 0.424 0.342 | 89.1% |
| y7_top1_lost | e_dso_clip12 | 6,651 | 1,623 | 0.393 | 0.037 | 1 | 0.372 0.402 0.422 0.429 0.340 | 89.1% |
| y7_top1_lost | e_dpo_proxy | 7,038 | 1,929 | 0.450 | 0.051 | -1 | 0.497 0.403 0.510 0.438 0.403 | 94.3% |
| y7_top1_lost | e_delay_coll | 5,158 | 1,304 | 0.574 | 0.041 | 1 | 0.569 0.572 0.518 0.633 0.576 | 69.1% |
| y7_top1_lost | e_ar_overdue | 6,859 | 1,912 | 0.605 | 0.048 | 1 | 0.582 0.668 0.540 0.630 0.603 | 91.9% |
| y7_top1_lost | e_ar_issued | 7,464 | 2,149 | 0.663 | 0.037 | -1 | 0.660 0.702 0.615 0.641 0.696 | 100.0% |
| y7_top1_lost | e_ar_issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 | 97.2% |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 | 100.0% |
| y7_top1_lost | log1p_a_in3 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 | 93.8% |


## 5. Honest leftover

Y3 leftover after days 0.474 (dies <0.55). Y7 leftover after issued_lag1 0.452 (dies). Y7 leftover after delay_coll 0.561 / both 0.565 (SHAP #1 is not just delay). Clip24 leftover Y3-days 0.549 Y7-iss 0.394 Y7-delay 0.404.

Leftover <0.55 dies. Y3 honest bar = days. Y7 honest bar = issued_lag1. Do not grow TURNOVER even if leftover lives.

| y | stem | residual | n | n_pos | CV | R² | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_dso_proxy | after issued_lag1 | 6,470 | 1,565 | 0.452 | 0.000 | 0.373 0.596 0.382 0.582 0.326 |
| y7_top1_lost | e_dso_proxy | after delay_coll | 4,735 | 1,047 | 0.561 | 0.000 | 0.547 0.508 0.491 0.642 0.617 |
| y7_top1_lost | e_dso_proxy | after issued_lag1+delay | 4,670 | 1,038 | 0.565 | 0.003 | 0.557 0.519 0.480 0.646 0.624 |
| y7_top1_lost | e_dso_proxy | after days | 6,651 | 1,623 | 0.553 | 0.000 | 0.627 0.604 0.596 0.525 0.412 |
| y7_top1_lost | e_dso_proxy | after size | 6,247 | 1,503 | 0.462 | 0.001 | 0.479 0.506 0.475 0.531 0.319 |
| y7_top1_lost | e_dso_proxy | after DPO | 6,389 | 1,536 | 0.473 | 0.003 | 0.415 0.584 0.419 0.514 0.432 |
| y7_top1_lost | e_dso_proxy | after ar_overdue | 6,216 | 1,490 | 0.524 | 0.000 | 0.447 0.577 0.466 0.573 0.558 |
| y7_top1_lost | e_dso_proxy | after issued | 6,651 | 1,623 | 0.502 | 0.000 | 0.621 0.611 0.374 0.577 0.325 |
| y7_top1_lost | e_dso_clip24 | after issued_lag1 | 6,470 | 1,565 | 0.394 | 0.003 | 0.364 0.401 0.435 0.427 0.342 |
| y7_top1_lost | e_dso_clip24 | after delay_coll | 4,735 | 1,047 | 0.404 | 0.053 | 0.411 0.368 0.434 0.493 0.313 |
| y7_top1_lost | e_dso_clip24 | after issued_lag1+delay | 4,670 | 1,038 | 0.408 | 0.056 | 0.408 0.373 0.451 0.494 0.312 |
| y7_top1_lost | e_dso_clip24 | after days | 6,651 | 1,623 | 0.429 | 0.000 | 0.370 0.602 0.417 0.425 0.330 |
| y7_top1_lost | e_dso_clip24 | after size | 6,247 | 1,503 | 0.423 | 0.001 | 0.365 0.593 0.413 0.423 0.322 |
| y7_top1_lost | e_dso_clip24 | after DPO | 6,389 | 1,536 | 0.392 | 0.004 | 0.370 0.389 0.434 0.433 0.336 |
| y7_top1_lost | e_dso_clip24 | after ar_overdue | 6,216 | 1,490 | 0.446 | 0.137 | 0.377 0.531 0.414 0.500 0.408 |
| y7_top1_lost | e_dso_clip24 | after issued | 6,651 | 1,623 | 0.432 | 0.000 | 0.369 0.601 0.419 0.423 0.346 |
| y3_recover_cash_6m | e_dso_proxy | after days | 2,418 | 93 | 0.474 | 0.000 | 0.494 0.722 0.424 0.431 0.299 |
| y3_recover_cash_6m | e_dso_proxy | after size | 2,376 | 90 | 0.632 | 0.001 | 0.512 0.567 0.736 0.662 0.684 |
| y3_recover_cash_6m | e_dso_proxy | after days+size | 2,376 | 90 | 0.618 | 0.001 | 0.490 0.546 0.739 0.646 0.671 |
| y3_recover_cash_6m | e_dso_proxy | after issued_lag1 | 2,418 | 93 | 0.560 | 0.000 | 0.665 0.382 0.579 0.701 0.473 |
| y3_recover_cash_6m | e_dso_proxy | after delay_coll | 1,638 | 65 | 0.428 | 0.000 | 0.370 0.405 0.369 0.498 0.495 |
| y3_recover_cash_6m | e_dso_clip24 | after days | 2,418 | 93 | 0.549 | 0.000 | 0.645 0.367 0.594 0.686 0.451 |
| y3_recover_cash_6m | e_dso_clip24 | after size | 2,376 | 90 | 0.564 | 0.001 | 0.683 0.394 0.582 0.690 0.472 |
| y3_recover_cash_6m | e_dso_clip24 | after days+size | 2,376 | 90 | 0.547 | 0.001 | 0.644 0.371 0.591 0.684 0.447 |
| y3_recover_cash_6m | e_dso_clip24 | after issued_lag1 | 2,418 | 93 | 0.565 | 0.003 | 0.677 0.381 0.585 0.700 0.483 |
| y3_recover_cash_6m | e_dso_clip24 | after delay_coll | 1,638 | 65 | 0.603 | 0.053 | 0.703 0.522 0.600 0.663 0.529 |


## 6. |DSO|>24 drop-tail leftover (DPO pattern)

Y3 leftover after days on drop>|24| 0.424 (raw leftover 0.474) clip12 0.556 after log1p(issued) 0.572. Y7 leftover-iss drop>|24| 0.418. Y3 leftover is not only the |DSO|>24 tail.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover days (raw) | 2,418 | 93 | 0.474 | 0.156 | -1 | 0.494 0.722 0.424 0.431 0.299 |
| y3_recover_cash_6m | leftover days drop |DSO|>24 | 2,267 | 79 | 0.424 | 0.071 | 1 | 0.510 0.376 0.410 0.342 0.481 |
| y3_recover_cash_6m | leftover days clip12 | 2,418 | 93 | 0.556 | 0.135 | 1 | 0.672 0.376 0.597 0.680 0.454 |
| y3_recover_cash_6m | leftover days clip24 | 2,418 | 93 | 0.549 | 0.135 | 1 | 0.645 0.367 0.594 0.686 0.451 |
| y3_recover_cash_6m | leftover after log1p(ar_issued) | 2,418 | 93 | 0.572 | 0.119 | -1 | 0.775 0.566 0.545 0.503 0.470 |
| y3_recover_cash_6m | leftover days+issued | 2,418 | 93 | 0.413 | 0.088 | -1 | 0.263 0.493 0.459 0.427 0.421 |
| y3_recover_cash_6m | DSO drop |x|>24 raw | 2,267 | 79 | 0.416 | 0.062 | 1 | 0.498 0.372 0.406 0.345 0.456 |
| y3_recover_cash_6m | has_|DSO|>24 | 5,648 | 402 | 0.494 | 0.017 | 1 | 0.469 0.495 0.498 0.515 0.493 |
| y3_recover_cash_6m | has_tiny_issued | 5,648 | 402 | 0.513 | 0.012 | -1 | 0.514 0.493 0.514 0.517 0.525 |
| y3_recover_cash_6m | has_dso flag | 5,648 | 402 | 0.597 | 0.048 | -1 | 0.565 0.556 0.564 0.646 0.654 |
| y7_top1_lost | leftover issued_lag1 (raw) | 6,470 | 1,565 | 0.452 | 0.127 | 1 | 0.373 0.596 0.382 0.582 0.326 |
| y7_top1_lost | leftover issued_lag1 drop |DSO|>24 | 6,018 | 1,371 | 0.418 | 0.067 | -1 | 0.407 0.450 0.468 0.459 0.306 |


## 7. Twin screen

TWIN |ρ|≥0.80 vs DPO / delay_coll / ar_overdue / issued: none. No twin drop.

## 8. Winsorise-at-24 (in-memory only)

Clip24 single Y3 0.564 Y7 0.391. Clip leftover Y3-days 0.549 (raw leftover 0.474) Y7-iss 0.394 (raw 0.452). Clip moves leftover ≥0.02 — still do not write the clip to parquet.

## 9. SIZE terciles and invoice-book-only (drop 470)

Book-only (drop 470) Y3 0.564 Y7 0.431. SIZE T1 Y3 — Y7 0.588 T3 Y3 — Y7 0.543.

| y | feature | n | n_pos | CV | sd | sign | folds | present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | DSO all | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 | 42.8% |
| y3_recover_cash_6m | DSO leftover-days all | 2,418 | 93 | 0.474 | 0.156 | -1 | 0.494 0.722 0.424 0.431 0.299 |  |
| y7_top1_lost | DSO all | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 | 89.1% |
| y7_top1_lost | DSO leftover-iss all | 6,470 | 1,565 | 0.452 | 0.127 | 1 | 0.373 0.596 0.382 0.582 0.326 |  |
| y3_recover_cash_6m | DSO book_only | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 | 66.8% |
| y3_recover_cash_6m | DSO leftover-days book_only | 2,418 | 93 | 0.474 | 0.156 | -1 | 0.494 0.722 0.424 0.431 0.299 |  |
| y7_top1_lost | DSO book_only | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 | 89.1% |
| y7_top1_lost | DSO leftover-iss book_only | 6,470 | 1,565 | 0.452 | 0.127 | 1 | 0.373 0.596 0.382 0.582 0.326 |  |
| y3_recover_cash_6m | DSO T1 | 376 | 33 | LOW_POWER | — | — | — | 32.5% |
| y3_recover_cash_6m | DSO leftover-days T1 | 376 | 33 | LOW_POWER | — | — | — |  |
| y7_top1_lost | DSO T1 | 1,512 | 382 | 0.588 | 0.158 | 1 | 0.758 0.580 0.598 0.669 0.334 | 84.4% |
| y7_top1_lost | DSO leftover-iss T1 | 1,512 | 382 | 0.586 | 0.155 | 1 | 0.752 0.568 0.604 0.669 0.338 |  |
| y3_recover_cash_6m | DSO T2 | 940 | 33 | LOW_POWER | — | — | — | 47.4% |
| y7_top1_lost | DSO T2 | 2,466 | 622 | 0.409 | 0.061 | -1 | 0.461 0.431 0.415 0.434 0.303 | 90.8% |
| y3_recover_cash_6m | DSO T3 | 1,060 | 24 | LOW_POWER | — | — | — | 44.4% |
| y3_recover_cash_6m | DSO leftover-days T3 | 1,060 | 24 | LOW_POWER | — | — | — |  |
| y7_top1_lost | DSO T3 | 2,269 | 499 | 0.543 | 0.108 | 1 | 0.640 0.613 0.574 0.524 0.367 | 91.1% |
| y7_top1_lost | DSO leftover-iss T3 | 2,269 | 499 | 0.555 | 0.120 | 1 | 0.621 0.613 0.646 0.543 0.351 |  |


## 10. Q6 — lag1/lag3; empty-on-short; exists early like DPO?

Q6 Y7 DSO now 0.431 lag1 0.401 lag3 0.396 short lag1 0.422. Early6 calendar DSO finite share 87.3% (exists earlier than delay). Delay early6 nn=0 (CONFIRM empty until month 7).

| slice | col | n_nn | n_pos | present | CV |
| --- | --- | --- | --- | --- | --- |
| all | e_dso_proxy | 6,651 | 1,623 | 89.1% | 0.431 |
| all | e_dso_proxy_lag1 | 6,338 | 1,594 | 84.9% | 0.401 |
| all | e_dso_proxy_lag3 | 5,317 | 1,328 | 71.2% | 0.396 |
| all | e_ar_issued_lag1 | 7,253 | 2,072 | 97.2% | 0.630 |
| all | e_delay_coll | 5,158 | 1,304 | 69.1% | 0.574 |
| all | e_dpo_proxy | 7,038 | 1,929 | 94.3% | 0.450 |
| short_<12_sofar | e_dso_proxy | 3,925 | 1,015 | 88.8% | 0.428 |
| short_<12_sofar | e_dso_proxy_lag1 | 3,617 | 961 | 81.8% | 0.422 |
| short_<12_sofar | e_dso_proxy_lag3 | 2,704 | 714 | 61.2% | 0.418 |
| short_<12_sofar | e_ar_issued_lag1 | 4,210 | 1,260 | 95.2% | 0.626 |
| short_<12_sofar | e_delay_coll | 2,535 | 641 | 57.3% | 0.581 |
| short_<12_sofar | e_dpo_proxy | 4,110 | 1,187 | 93.0% | 0.451 |
| long_>=18_sofar | e_dso_proxy | 828 | 157 | 90.4% | 0.607 |
| long_>=18_sofar | e_dso_proxy_lag1 | 829 | 163 | 90.5% | 0.596 |
| long_>=18_sofar | e_dso_proxy_lag3 | 795 | 156 | 86.8% | 0.567 |
| long_>=18_sofar | e_ar_issued_lag1 | 916 | 209 | 100.0% | 0.646 |
| long_>=18_sofar | e_delay_coll | 806 | 168 | 88.0% | 0.561 |
| long_>=18_sofar | e_dpo_proxy | 900 | 199 | 98.3% | 0.591 |
| short_<12_company | e_dso_proxy | 704 | 127 | 91.0% | 0.634 |
| short_<12_company | e_dso_proxy_lag1 | 554 | 90 | 71.6% | 0.605 |
| short_<12_company | e_dso_proxy_lag3 | 264 | 33 | 34.1% | LOW_POWER |
| short_<12_company | e_ar_issued_lag1 | 666 | 140 | 86.0% | 0.722 |
| short_<12_company | e_delay_coll | 477 | 59 | 61.6% | 0.567 |
| short_<12_company | e_dpo_proxy | 711 | 150 | 91.9% | 0.625 |
| long_>=18_company | e_dso_proxy | 5,539 | 1,408 | 88.9% | 0.385 |
| long_>=18_company | e_dso_proxy_lag1 | 5,417 | 1,421 | 87.0% | 0.394 |
| long_>=18_company | e_dso_proxy_lag3 | 4,775 | 1,225 | 76.6% | 0.395 |
| long_>=18_company | e_ar_issued_lag1 | 6,154 | 1,813 | 98.8% | 0.616 |
| long_>=18_company | e_delay_coll | 4,349 | 1,164 | 69.8% | 0.573 |
| long_>=18_company | e_dpo_proxy | 5,920 | 1,667 | 95.0% | 0.452 |
| early6_calendar | e_dso_proxy | 935 | 299 | 87.3% | 0.536 |
| early6_calendar | e_dso_proxy_lag1 | 874 | 281 | 81.6% | 0.538 |
| early6_calendar | e_dso_proxy_lag3 | 545 | 167 | 50.9% | 0.579 |
| early6_calendar | e_ar_issued_lag1 | 1,002 | 357 | 93.6% | 0.588 |
| early6_calendar | e_delay_coll | 0 | 0 | 0.0% | LOW_POWER |
| early6_calendar | e_dpo_proxy | 1,003 | 350 | 93.7% | 0.400 |
| after_month7 | e_dso_proxy | 5,716 | 1,324 | 89.4% | 0.380 |
| after_month7 | e_dso_proxy_lag1 | 5,464 | 1,313 | 85.5% | 0.390 |
| after_month7 | e_dso_proxy_lag3 | 4,772 | 1,161 | 74.6% | 0.384 |
| after_month7 | e_ar_issued_lag1 | 6,251 | 1,715 | 97.8% | 0.643 |
| after_month7 | e_delay_coll | 5,158 | 1,304 | 80.7% | 0.574 |
| after_month7 | e_dpo_proxy | 6,035 | 1,579 | 94.4% | 0.445 |


## 11. ICC / company-demean

DSO ICC=0.628 k=673 (DPO ICC=0.807 locked 0.807). Y7 raw 0.431 demean 0.475 company-mean 0.489. acf1=0.255 (trait-ish / borderline). not BETWEEN (≥0.85) — company-mean carries skill.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_dso_proxy raw | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 |
| y7_top1_lost | e_dso_proxy demean | 6,651 | 1,623 | 0.475 | 0.035 | 1 | 0.430 0.489 0.523 0.479 0.453 |
| y7_top1_lost | e_dso_proxy company-mean | 7,462 | 2,147 | 0.489 | 0.133 | 1 | 0.341 0.620 0.558 0.576 0.352 |
| y3_recover_cash_6m | e_dso_proxy raw | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 |
| y3_recover_cash_6m | e_dso_proxy demean | 2,418 | 93 | 0.541 | 0.077 | 1 | 0.610 0.601 0.418 0.544 0.532 |
| y3_recover_cash_6m | e_dso_proxy company-mean | 3,398 | 205 | 0.396 | 0.038 | -1 | 0.416 0.349 0.446 0.394 0.373 |


## 12. Fold 4 / short-DSO quintile

Fold 4 Y7 DSO 0.342 (quote 0.342) vs issued_lag1 0.647 (quote 0.647) vs TURNOVER 0.680. CONFIRM issued owns fold 4. Short-DSO Q1 n=1331 (B_shallow card 1331 / OOF 0.410): univariate DSO 0.455 issued 0.604. CONFIRM short-DSO hole (near locked 0.410).

| feature | CV | fold4 | folds |
| --- | --- | --- | --- |
| e_dso_proxy | 0.431 | 0.342 | 0.370 0.600 0.420 0.424 0.342 |
| e_dso_clip24 | 0.391 | 0.342 | 0.371 0.400 0.421 0.424 0.342 |
| e_ar_issued_lag1 | 0.630 | 0.647 | 0.643 0.662 0.590 0.605 0.647 |
| e_delay_coll | 0.574 | 0.576 | 0.569 0.572 0.518 0.633 0.576 |
| e_dpo_proxy | 0.450 | 0.403 | 0.497 0.403 0.510 0.438 0.403 |
| c_n_days_with_tx | 0.458 | 0.439 | 0.457 0.483 0.478 0.432 0.439 |


| DSO q | n | rate | DSO p50 | delay nn | DSO CV | issued CV | delay CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | 1331 | 25.5% | 0.06 | 80.5% | 0.455 | 0.604 | 0.516 |
| Q2 | 1330 | 25.0% | 1.00 | 68.2% | 0.449 | 0.647 | 0.601 |
| Q3 | 1330 | 22.1% | 1.74 | 75.8% | 0.582 | 0.448 | 0.604 |
| Q4 | 1330 | 19.8% | 3.74 | 70.6% | 0.546 | 0.521 | 0.536 |
| Q5 | 1330 | 29.6% | 15.45 | 60.8% | 0.644 | 0.644 | 0.527 |


## 13. vs delay leftover KEEP — leftover of DSO after delay_coll

Same-n DSO+delay: raw DSO 0.370 leftover-after-delay 0.561 R²=0.000. Raw delay 0.579 leftover-after-DSO 0.581 (locked KEEP 0.581, R²=0.000). SHAP #1 is not just delay (leftover of DSO after delay lives or DSO itself is already <0.55). Delay leftover after DSO CONFIRM KEEP.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | DSO same-n delay | 4,735 | 1,047 | 0.370 | 0.051 | -1 | 0.380 0.321 0.430 0.405 0.316 |
| y7_top1_lost | delay same-n DSO | 4,735 | 1,047 | 0.579 | 0.045 | 1 | 0.576 0.560 0.520 0.644 0.594 |
| y7_top1_lost | DSO leftover after delay | 4,735 | 1,047 | 0.561 | 0.066 | -1 | 0.547 0.508 0.491 0.642 0.617 |
| y7_top1_lost | delay leftover after DSO | 4,735 | 1,047 | 0.581 | 0.049 | 1 | 0.577 0.554 0.519 0.643 0.614 |
| y7_top1_lost | issued_lag1 same-n | 4,670 | 1,038 | 0.578 | 0.064 | -1 | 0.601 0.523 0.506 0.599 0.662 |


## Extra — holdout coverage only (no AUROC)

Holdout 72 coverage only: 1,073 CM / 72 companies. Y7 pos=122 (quote 122). DSO cov 41.9%; early6 25.5%; dark nn=0. No AUROC claim.

| col | n_cm | nn | cov | early6 nn | dark nn | |DSO|>24 |
| --- | --- | --- | --- | --- | --- | --- |
| e_dso_proxy | 1,073 | 450 | 41.9% | 25.5% | 0 | 3.8% |


## Extra — same-n leftover after issued_lag1

Same-n (DSO+issued_lag1 finite): raw DSO 0.429 leftover 0.452 issued 0.589 R²=0.000. leftover differs from raw.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | DSO same-n | 6,470 | 1,565 | 0.429 | 0.101 | 1 | 0.366 0.598 0.420 0.424 0.339 |
| y7_top1_lost | issued_lag1 same-n | 6,470 | 1,565 | 0.589 | 0.041 | -1 | 0.604 0.578 0.543 0.569 0.651 |
| y7_top1_lost | leftover iss same-n | 6,470 | 1,565 | 0.452 | 0.127 | 1 | 0.373 0.596 0.382 0.582 0.326 |
| y7_top1_lost | clip24 same-n | 6,470 | 1,565 | 0.429 | 0.101 | 1 | 0.367 0.597 0.421 0.424 0.338 |
| y7_top1_lost | delay same-n | 4,670 | 1,038 | 0.579 | 0.047 | 1 | 0.581 0.558 0.518 0.644 0.595 |


## Extra — quintiles of DSO vs Y3 / Y7

DSO quintiles vs Y3/Y7. no clear Q5 tail on Y3.

| y | DSO q | n | n_pos | rate | DSO p50 | issued p50 | share>24 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | Q1 | 484 | 16 | 3.3% | 0.01 | 149847 | 0.0% |
| y3_recover_cash_6m | Q2 | 483 | 23 | 4.8% | 1.00 | 111252 | 0.0% |
| y3_recover_cash_6m | Q3 | 484 | 12 | 2.5% | 1.78 | 177868 | 0.0% |
| y3_recover_cash_6m | Q4 | 483 | 10 | 2.1% | 3.92 | 150964 | 0.0% |
| y3_recover_cash_6m | Q5 | 484 | 32 | 6.6% | 13.35 | 31770 | 31.2% |
| y7_top1_lost | Q1 | 1331 | 340 | 25.5% | 0.06 | 88098 | 0.0% |
| y7_top1_lost | Q2 | 1330 | 332 | 25.0% | 1.00 | 79056 | 0.0% |
| y7_top1_lost | Q3 | 1330 | 294 | 22.1% | 1.74 | 168087 | 0.0% |
| y7_top1_lost | Q4 | 1330 | 263 | 19.8% | 3.74 | 129168 | 0.0% |
| y7_top1_lost | Q5 | 1330 | 394 | 29.6% | 15.45 | 34750 | 35.0% |


## Extra — tiny-issued blow-up

Tiny-issued p10=3756: DSO p50 on tiny 2.99 share>24 28.9%. Y7 DSO on tiny 0.600 vs rest 0.398.

| slice | n | DSO nn | DSO p50 | DSO p99 | |DSO|>24 | issued p50 |
| --- | --- | --- | --- | --- | --- | --- |
| tiny_issued p10 | 859 | 859 | 2.99 | 38933.1 | 28.9% | 1089 |
| rest issued>p10 | 7724 | 7724 | 1.70 | 127.2 | 4.8% | 128971 |
| issued=0 / NaN DSO | 12574 | 0 | — | — | — | 0 |


## Extra — Y3 same-n leftover vs days

Y3 same-n raw 0.564 leftover-days 0.474 days 0.689 issued 0.593 R²=0.000.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | DSO same-n | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 |
| y3_recover_cash_6m | days same-n | 2,418 | 93 | 0.689 | 0.103 | -1 | 0.628 0.723 0.544 0.805 0.746 |
| y3_recover_cash_6m | issued_lag1 same-n | 2,418 | 93 | 0.593 | 0.070 | -1 | 0.685 0.613 0.555 0.498 0.615 |
| y3_recover_cash_6m | leftover days same-n | 2,418 | 93 | 0.474 | 0.156 | -1 | 0.494 0.722 0.424 0.431 0.299 |
| y3_recover_cash_6m | clip24 same-n | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.382 0.587 0.697 0.475 |
| y3_recover_cash_6m | size same-n | 2,376 | 90 | 0.615 | 0.079 | -1 | 0.503 0.577 0.703 0.622 0.672 |


## Extra — fold-4 problem groups

Fold-4 problem groups DSO p50 vs rest (table). Y7 DSO on problem groups 0.433 vs rest 0.591.

| slice | n_lab | n_pos | rate | DSO nn | DSO p50 | issued p50 |
| --- | --- | --- | --- | --- | --- | --- |
| fold4 problem groups | 540 | 371 | 68.7% | 84.8% | 0.18 | 28367 |
| rest labeled | 6924 | 1778 | 25.7% | 89.4% | 1.97 | 66700 |


## Extra — ICC on clip24

ICC raw 0.628 clip24 0.912. acf1 clip24 0.292 (raw 0.255). clip makes BETWEEN.

## Extra — Y3 Q6 lag / early leftover

Y3 Q6 DSO lag1 0.588 short 0.602. After-month7 leftover-days 0.477.

| slice | col | n_nn | n_pos | present | CV |
| --- | --- | --- | --- | --- | --- |
| all | e_dso_proxy | 2,418 | 93 | 42.8% | 0.564 |
| all | e_dso_proxy_lag1 | 2,385 | 100 | 42.2% | 0.588 |
| all | e_dso_proxy_lag3 | 2,114 | 103 | 37.4% | 0.628 |
| all | c_n_days_with_tx | 5,648 | 402 | 100.0% | 0.711 |
| all | leftover days | 2,418 | 93 | 42.8% | 0.474 |
| short_<12_sofar | e_dso_proxy | 1,528 | 55 | 41.0% | 0.590 |
| short_<12_sofar | e_dso_proxy_lag1 | 1,485 | 57 | 39.9% | 0.602 |
| short_<12_sofar | e_dso_proxy_lag3 | 1,221 | 54 | 32.8% | 0.605 |
| short_<12_sofar | c_n_days_with_tx | 3,723 | 252 | 100.0% | 0.696 |
| short_<12_sofar | leftover days | 1,528 | 55 | 41.0% | 0.546 |
| early6_calendar | e_dso_proxy | 434 | 11 | 43.0% | LOW_POWER |
| early6_calendar | e_dso_proxy_lag1 | 425 | 11 | 42.1% | LOW_POWER |
| early6_calendar | e_dso_proxy_lag3 | 272 | 12 | 27.0% | LOW_POWER |
| early6_calendar | c_n_days_with_tx | 1,009 | 45 | 100.0% | LOW_POWER |
| early6_calendar | leftover days | 434 | 11 | 43.0% | LOW_POWER |
| after_month7 | e_dso_proxy | 1,984 | 82 | 42.8% | 0.582 |
| after_month7 | e_dso_proxy_lag1 | 1,960 | 89 | 42.3% | 0.604 |
| after_month7 | e_dso_proxy_lag3 | 1,842 | 91 | 39.7% | 0.633 |
| after_month7 | c_n_days_with_tx | 4,639 | 357 | 100.0% | 0.722 |
| after_month7 | leftover days | 1,984 | 82 | 42.8% | 0.477 |


## Extra — Y3/Y7 lag1 leftover after days

Y3 lag1 leftover after days 0.711 drop>24 0.553 (now leftover 0.474).

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | lag1 raw | 2,385 | 100 | 0.588 | 0.109 | 1 | 0.653 0.436 0.606 0.715 0.528 |
| y3_recover_cash_6m | lag1 leftover days | 2,385 | 100 | 0.711 | 0.108 | 1 | 0.690 0.729 0.544 0.839 0.755 |
| y3_recover_cash_6m | lag1 leftover days drop>24 | 2,248 | 87 | 0.553 | 0.099 | 1 | 0.505 0.464 0.614 0.696 0.485 |
| y3_recover_cash_6m | now leftover days | 2,418 | 93 | 0.474 | 0.156 | -1 | 0.494 0.722 0.424 0.431 0.299 |
| y7_top1_lost | lag1 raw | 6,338 | 1,594 | 0.401 | 0.059 | -1 | 0.344 0.433 0.436 0.460 0.330 |
| y7_top1_lost | lag1 leftover days | 6,338 | 1,594 | 0.457 | 0.014 | 1 | 0.460 0.469 0.469 0.441 0.444 |
| y7_top1_lost | lag1 leftover days drop>24 | 5,909 | 1,422 | 0.415 | 0.083 | -1 | 0.368 0.472 0.463 0.481 0.291 |
| y7_top1_lost | now leftover days | 6,651 | 1,623 | 0.553 | 0.088 | 1 | 0.627 0.604 0.596 0.525 0.412 |


## Extra — Y7 leftover-after-delay is a tail?

Y7 leftover after delay raw 0.561 drop>|24| 0.424 clip24 0.404 clip12 0.399 R² raw=0.000 drop=0.061 clip=0.053. 0.561 leftover-after-delay is a |DSO|>24 tail (clip/drop dies).

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover delay (raw) | 4,735 | 1,047 | 0.561 | 0.066 | -1 | 0.547 0.508 0.491 0.642 0.617 |
| y7_top1_lost | leftover delay drop |DSO|>24 | 4,458 | 934 | 0.424 | 0.087 | -1 | 0.453 0.411 0.466 0.509 0.281 |
| y7_top1_lost | leftover delay clip24 | 4,735 | 1,047 | 0.404 | 0.068 | -1 | 0.411 0.368 0.434 0.493 0.313 |
| y7_top1_lost | leftover delay clip12 | 4,735 | 1,047 | 0.399 | 0.065 | -1 | 0.409 0.362 0.436 0.477 0.310 |
| y7_top1_lost | leftover delay+issued | 4,670 | 1,038 | 0.565 | 0.070 | -1 | 0.557 0.519 0.480 0.646 0.624 |
| y7_top1_lost | DSO raw on delay-overlap | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 |
| y7_top1_lost | delay raw on overlap | 5,158 | 1,304 | 0.574 | 0.041 | 1 | 0.569 0.572 0.518 0.633 0.576 |


## Extra — Y3 leftover days vs days+size

Y3 leftover days 0.474 size 0.632 days+size 0.618 days+size drop>24 0.418 R² days=0.000 size=0.001 both=0.001. Same-n days 0.695 leftover-days 0.471.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover days all | 2,418 | 93 | 0.474 | 0.156 | -1 | 0.494 0.722 0.424 0.431 0.299 |
| y3_recover_cash_6m | leftover size all | 2,376 | 90 | 0.632 | 0.091 | 1 | 0.512 0.567 0.736 0.662 0.684 |
| y3_recover_cash_6m | leftover days+size all | 2,376 | 90 | 0.618 | 0.099 | 1 | 0.490 0.546 0.739 0.646 0.671 |
| y3_recover_cash_6m | leftover days+size drop>24 | 2,230 | 76 | 0.418 | 0.062 | 1 | 0.505 0.385 0.398 0.349 0.455 |
| y3_recover_cash_6m | leftover days drop>24 | 2,267 | 79 | 0.424 | 0.071 | 1 | 0.510 0.376 0.410 0.342 0.481 |
| y3_recover_cash_6m | days same-n three | 2,376 | 90 | 0.695 | 0.096 | -1 | 0.629 0.733 0.563 0.799 0.750 |
| y3_recover_cash_6m | size same-n three | 2,376 | 90 | 0.615 | 0.079 | -1 | 0.503 0.577 0.703 0.622 0.672 |
| y3_recover_cash_6m | DSO same-n three | 2,376 | 90 | 0.570 | 0.133 | 1 | 0.682 0.389 0.603 0.695 0.478 |
| y3_recover_cash_6m | leftover days same-n three | 2,376 | 90 | 0.471 | 0.159 | -1 | 0.495 0.724 0.424 0.415 0.296 |
| y3_recover_cash_6m | leftover days+size same-n three | 2,376 | 90 | 0.618 | 0.099 | 1 | 0.490 0.546 0.739 0.646 0.671 |


## Extra — Y3 lag1 leftover vs days clone

Y3 lag1 leftover vs days ρ=-0.984 n=8080 R²=0.000 (raw lag1 vs days 0.023; FALSE clone — unclipped residual rank artifact). Same-n leftover 0.711 days 0.707 lag1 raw 0.588.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | lag1 raw | 2,385 | 100 | 0.588 | 0.109 | 1 | 0.653 0.436 0.606 0.715 0.528 |
| y3_recover_cash_6m | days same-n lag1 | 2,385 | 100 | 0.707 | 0.102 | -1 | 0.682 0.736 0.544 0.810 0.761 |
| y3_recover_cash_6m | lag1 leftover days | 2,385 | 100 | 0.711 | 0.108 | 1 | 0.690 0.729 0.544 0.839 0.755 |
| y3_recover_cash_6m | |leftover| | 2,385 | 100 | 0.690 | 0.080 | -1 | 0.692 0.736 0.551 0.724 0.746 |
| y3_recover_cash_6m | days on leftover-defined | 2,385 | 100 | 0.707 | 0.102 | -1 | 0.682 0.736 0.544 0.810 0.761 |


## Extra — fat issued (issued > p10) leftover

Fat issued>p10=3756: Y3 DSO 0.479 leftover-days 0.460 vs days 0.684. Y7 leftover-iss 0.385.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | fat issued DSO | 2,166 | 85 | 0.479 | 0.146 | 1 | 0.690 0.428 0.546 0.303 0.429 |
| y3_recover_cash_6m | fat leftover days | 2,166 | 85 | 0.460 | 0.164 | -1 | 0.484 0.684 0.470 0.442 0.223 |
| y3_recover_cash_6m | fat days | 2,166 | 85 | 0.684 | 0.099 | -1 | 0.622 0.705 0.551 0.801 0.741 |
| y7_top1_lost | fat issued DSO | 5,957 | 1,349 | 0.398 | 0.054 | -1 | 0.399 0.396 0.446 0.439 0.311 |
| y7_top1_lost | fat leftover iss | 5,805 | 1,308 | 0.385 | 0.052 | -1 | 0.401 0.401 0.397 0.431 0.294 |
| y7_top1_lost | fat issued_lag1 | 5,805 | 1,308 | 0.559 | 0.063 | -1 | 0.549 0.528 0.501 0.551 0.665 |


## Extra — fold-wise leftover

Y7 fold-wise leftover-iss 0.452 folds 0.373 0.596 0.382 0.582 0.326 fold4 0.326 vs issued fold4 0.647.

| feature | CV | folds | fold4 |
| --- | --- | --- | --- |
| e_dso_proxy | 0.431 | 0.370 0.600 0.420 0.424 0.342 | 0.342 |
| e_ar_issued_lag1 | 0.630 | 0.643 0.662 0.590 0.605 0.647 | 0.647 |
| leftover iss | 0.452 | 0.373 0.596 0.382 0.582 0.326 | 0.326 |
| leftover delay | 0.561 | 0.547 0.508 0.491 0.642 0.617 | 0.617 |
| Y3 leftover days | 0.474 | 0.494 0.722 0.424 0.431 0.299 | 0.299 |
| Y3 days | 0.711 | 0.665 0.738 0.700 0.715 0.740 | 0.740 |


## Extra — chronic Y2 (Y3 only)

Chronic Y2 (rate≥0.5) companies: 85. Y3 DSO 0.564 → drop 0.566. Days 0.711 → 0.708. does not flip.

| slice | DSO | days |
| --- | --- | --- |
| Y3 all | 0.564 | 0.711 |
| Y3 drop-chronic n_co=85 | 0.566 | 0.708 |


## Extra — log1p(DSO) leftover (in-memory)

log1p(DSO) Y3 0.564 leftover-days 0.560 R²=0.000. Y7 log1p leftover-iss 0.431 leftover-delay 0.389 R²iss=0.001.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | log1p DSO | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.477 |
| y3_recover_cash_6m | log1p leftover days | 2,418 | 93 | 0.560 | 0.136 | 1 | 0.670 0.378 0.595 0.693 0.463 |
| y3_recover_cash_6m | clip24 leftover days | 2,418 | 93 | 0.549 | 0.135 | 1 | 0.645 0.367 0.594 0.686 0.451 |
| y7_top1_lost | log1p DSO | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 |
| y7_top1_lost | log1p leftover iss | 6,470 | 1,565 | 0.431 | 0.101 | 1 | 0.364 0.600 0.425 0.426 0.341 |
| y7_top1_lost | log1p leftover delay | 4,735 | 1,047 | 0.389 | 0.058 | -1 | 0.403 0.348 0.427 0.454 0.311 |
| y7_top1_lost | clip24 leftover iss | 6,470 | 1,565 | 0.394 | 0.040 | 1 | 0.364 0.401 0.435 0.427 0.342 |


## Extra — DSO vs AR open

DSO vs open ρ=0.600 n=8583 (not a twin — locked 0.600). vs issued -0.048. Y7 open 0.465 open-leftover-issued 0.596 DSO 0.431 R²=0.422.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_dso_proxy | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 |
| y7_top1_lost | e_ar_open | 7,464 | 2,149 | 0.465 | 0.092 | -1 | 0.508 0.514 0.493 0.511 0.301 |
| y7_top1_lost | e_ar_issued | 7,464 | 2,149 | 0.663 | 0.037 | -1 | 0.660 0.702 0.615 0.641 0.696 |
| y7_top1_lost | open leftover issued | 7,464 | 2,149 | 0.596 | 0.058 | 1 | 0.647 0.668 0.566 0.539 0.558 |


## Extra — lag1 Spearman vs days (clone check)

Spearman DSO-now vs days 0.022 lag1 vs days 0.023 clip24-lag1 vs days 0.023. Y3 clip24-lag1 leftover-days 0.569 raw clip24-lag1 0.587 R²=0.000.

| pair | ρ | n | twin? |
| --- | --- | --- | --- |
| DSO now vs days | 0.022 | 8,583 |  |
| DSO lag1 vs days | 0.023 | 8,080 |  |
| clip24 lag1 vs days | 0.023 | 8,080 |  |
| DSO lag1 vs DSO now | 0.796 | 7,464 |  |
| clip24 lag1 vs clip24 now | 0.797 | 7,464 |  |


## Extra — Y3 leftover after issued_lag1 tail

Y3 leftover after issued_lag1 raw 0.560 drop>24 0.416 clip24 0.565. 0.560 is a tail.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover iss raw | 2,418 | 93 | 0.560 | 0.133 | 1 | 0.665 0.382 0.579 0.701 0.473 |
| y3_recover_cash_6m | leftover iss drop>24 | 2,267 | 79 | 0.416 | 0.063 | 1 | 0.493 0.372 0.406 0.342 0.464 |
| y3_recover_cash_6m | leftover iss clip24 | 2,418 | 93 | 0.565 | 0.134 | 1 | 0.677 0.381 0.585 0.700 0.483 |
| y3_recover_cash_6m | DSO drop>24 raw | 2,267 | 79 | 0.416 | 0.062 | 1 | 0.498 0.372 0.406 0.345 0.456 |


## Extra — extreme DSO rows (train)

Max DSO=2190260.7 on train. n>1000=70 n>1e5=5. Top rows are tiny-issued / fat-open (table). Do not rewrite the store.

| company | period | DSO | AR open | AR issued | |DSO|>24 |
| --- | --- | --- | --- | --- | --- |
| COMP_0813 | 2026-03-01 | 2190260.7 | 5563262 | 2.5 | yes |
| COMP_0004 | 2025-12-01 | 1829450.8 | 585424 | 0.3 | yes |
| COMP_0909 | 2025-06-01 | 1675893.4 | 25138401 | 15.0 | yes |
| COMP_0460 | 2026-01-01 | 619117.5 | 3912823 | 6.3 | yes |
| COMP_0342 | 2025-08-01 | 337328.4 | 4081674 | 12.1 | yes |
| COMP_1046 | 2025-10-01 | 98205.3 | 26515442 | 270.0 | yes |
| COMP_1046 | 2025-12-01 | 48697.9 | 26515986 | 544.5 | yes |
| COMP_1046 | 2025-09-01 | 47957.4 | 26515172 | 552.9 | yes |


## Extra — demean leftover after issued / delay

Y7 demean leftover-iss 0.461 mean leftover-iss 0.518 demean leftover-delay 0.581.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | demean leftover iss | 6,470 | 1,565 | 0.461 | 0.030 | 1 | 0.427 0.496 0.472 0.475 0.433 |
| y7_top1_lost | mean leftover iss | 7,253 | 2,072 | 0.518 | 0.143 | 1 | 0.666 0.614 0.399 0.575 0.337 |
| y7_top1_lost | demean leftover delay | 4,735 | 1,047 | 0.581 | 0.053 | 1 | 0.566 0.572 0.505 0.642 0.617 |
| y7_top1_lost | demean raw | 6,651 | 1,623 | 0.475 | 0.035 | 1 | 0.430 0.489 0.523 0.479 0.453 |
| y7_top1_lost | company-mean raw | 7,462 | 2,147 | 0.489 | 0.133 | 1 | 0.341 0.620 0.558 0.576 0.352 |


## Extra — Y3 fold-wise leftover after days

Y3 fold-wise leftover-days 0.474 folds 0.494 0.722 0.424 0.431 0.299 vs days 0.665 0.738 0.700 0.715 0.740.

| feature | CV | folds | fold4 |
| --- | --- | --- | --- |
| e_dso_proxy | 0.564 | 0.680 0.380 0.588 0.697 0.476 | 0.476 |
| c_n_days_with_tx | 0.711 | 0.665 0.738 0.700 0.715 0.740 | 0.740 |
| leftover days | 0.474 | 0.494 0.722 0.424 0.431 0.299 | 0.299 |
| e_dso_clip24 | 0.564 | 0.680 0.382 0.587 0.697 0.475 | 0.475 |


## Extra — short-DSO Q1 leftover

Short-DSO Q1 leftover-iss 0.472 leftover-delay 0.521 vs issued 0.604 DSO 0.455.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | DSO Q1 | 1,331 | 340 | 0.455 | 0.069 | -1 | 0.466 0.499 0.427 0.353 0.532 |
| y7_top1_lost | issued Q1 | 1,298 | 328 | 0.604 | 0.046 | -1 | 0.621 0.580 0.537 0.625 0.655 |
| y7_top1_lost | delay Q1 | 1,072 | 283 | 0.516 | 0.177 | 1 | 0.591 0.780 0.387 0.484 0.337 |
| y7_top1_lost | leftover iss Q1 | 1,298 | 328 | 0.472 | 0.060 | -1 | 0.543 0.504 0.402 0.419 0.491 |
| y7_top1_lost | leftover delay Q1 | 1,072 | 283 | 0.521 | 0.184 | -1 | 0.595 0.780 0.406 0.524 0.299 |


## Extra — after month 7 leftover

After month 7: Y7 leftover-delay 0.561 drop 0.424 leftover-iss 0.372. Y3 leftover-iss 0.575.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | DSO after7 | 5,716 | 1,324 | 0.380 | 0.032 | 1 | 0.382 0.362 0.415 0.404 0.337 |
| y7_top1_lost | leftover delay after7 | 4,735 | 1,047 | 0.561 | 0.066 | -1 | 0.547 0.508 0.491 0.642 0.617 |
| y7_top1_lost | leftover delay drop after7 | 4,458 | 934 | 0.424 | 0.087 | -1 | 0.453 0.411 0.466 0.509 0.281 |
| y7_top1_lost | leftover iss after7 | 5,594 | 1,298 | 0.372 | 0.035 | 1 | 0.384 0.367 0.378 0.412 0.317 |
| y3_recover_cash_6m | DSO after7 | 1,984 | 82 | 0.582 | 0.127 | 1 | 0.724 0.419 0.613 0.669 0.484 |
| y3_recover_cash_6m | leftover delay after7 | 1,638 | 65 | 0.428 | 0.065 | -1 | 0.370 0.405 0.369 0.498 0.495 |
| y3_recover_cash_6m | leftover delay drop after7 | 1,554 | 56 | 0.556 | 0.066 | 1 | 0.538 0.490 0.594 0.650 0.505 |
| y3_recover_cash_6m | leftover iss after7 | 1,984 | 82 | 0.575 | 0.123 | 1 | 0.705 0.420 0.601 0.672 0.480 |


## Extra — holdout tails (coverage only)

Holdout tails (coverage only): nn=450 p50=2.44 p99=149.8 |DSO|>24 3.8%. No AUROC.

| slice | n_cm | nn | p50 | p99 | max | |DSO|>24 |
| --- | --- | --- | --- | --- | --- | --- |
| holdout all | 1,073 | 450 | 2.44 | 149.8 | 6293.3 | 3.8% |


## Extra — long-trail leftover after issued

Long so-far Y7 DSO 0.607 leftover-iss 0.613 issued 0.646 drop>24 0.556. Short so-far leftover-iss 0.503.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | DSO long_>=18_sofar | 828 | 157 | 0.607 | 0.124 | 1 | 0.758 0.615 0.655 0.593 0.416 |
| y7_top1_lost | issued_lag1 long_>=18_sofar | 916 | 209 | 0.646 | 0.077 | -1 | 0.631 0.601 0.552 0.742 0.705 |
| y7_top1_lost | leftover iss long_>=18_sofar | 828 | 157 | 0.613 | 0.131 | 1 | 0.755 0.618 0.700 0.579 0.415 |
| y7_top1_lost | leftover iss drop>24 long_>=18_sofar | 747 | 124 | 0.556 | 0.133 | 1 | 0.735 0.545 0.588 0.548 0.363 |
| y7_top1_lost | leftover delay long_>=18_sofar | 742 | 133 | 0.548 | 0.052 | -1 | 0.557 0.579 0.470 0.604 0.528 |
| y7_top1_lost | DSO short_<12_sofar | 3,925 | 1,015 | 0.428 | 0.070 | 1 | 0.366 0.542 0.431 0.426 0.374 |
| y7_top1_lost | issued_lag1 short_<12_sofar | 4,210 | 1,260 | 0.626 | 0.068 | -1 | 0.650 0.703 0.615 0.519 0.645 |
| y7_top1_lost | leftover iss short_<12_sofar | 3,744 | 957 | 0.503 | 0.120 | 1 | 0.632 0.534 0.405 0.591 0.352 |
| y7_top1_lost | leftover iss drop>24 short_<12_sofar | 3,505 | 845 | 0.430 | 0.065 | -1 | 0.408 0.485 0.470 0.461 0.325 |
| y7_top1_lost | leftover delay short_<12_sofar | 2,329 | 513 | 0.569 | 0.058 | -1 | 0.557 0.509 0.522 0.614 0.641 |
| y7_top1_lost | DSO long_>=18_company | 5,539 | 1,408 | 0.385 | 0.056 | -1 | 0.349 0.434 0.408 0.429 0.306 |
| y7_top1_lost | issued_lag1 long_>=18_company | 6,154 | 1,813 | 0.616 | 0.038 | -1 | 0.633 0.648 0.558 0.597 0.643 |
| y7_top1_lost | leftover iss long_>=18_company | 5,474 | 1,375 | 0.399 | 0.104 | 1 | 0.355 0.568 0.356 0.421 0.296 |
| y7_top1_lost | leftover iss drop>24 long_>=18_company | 5,116 | 1,220 | 0.414 | 0.080 | -1 | 0.381 0.477 0.456 0.466 0.287 |
| y7_top1_lost | leftover delay long_>=18_company | 3,976 | 938 | 0.564 | 0.073 | -1 | 0.552 0.512 0.480 0.656 0.621 |
| y7_top1_lost | DSO short_<12_company | 704 | 127 | 0.634 | 0.130 | 1 | 0.440 0.765 0.633 0.739 0.594 |
| y7_top1_lost | issued_lag1 short_<12_company | 666 | 140 | 0.722 | 0.050 | -1 | 0.719 0.662 0.748 0.791 0.692 |
| y7_top1_lost | leftover iss short_<12_company | 613 | 107 | 0.605 | 0.135 | 1 | 0.426 0.754 0.558 0.727 0.561 |
| y7_top1_lost | leftover iss drop>24 short_<12_company | 577 | 91 | 0.608 | 0.166 | 1 | 0.387 0.772 0.589 0.771 0.519 |
| y7_top1_lost | leftover delay short_<12_company | 458 | 50 | 0.668 | 0.147 | 1 | 0.562 0.767 0.622 0.870 0.519 |


## Extra — leftover after ar_overdue_30

Y7 leftover after ar_overdue_30 0.477 clip24 0.549 od30 0.580 R²=0.001. Y3 leftover-od30 0.374.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover od30 | 6,216 | 1,490 | 0.477 | 0.031 | 1 | 0.500 0.467 0.517 0.455 0.446 |
| y7_top1_lost | clip24 leftover od30 | 6,216 | 1,490 | 0.549 | 0.044 | 1 | 0.592 0.557 0.590 0.507 0.501 |
| y7_top1_lost | e_ar_overdue_30 | 6,859 | 1,912 | 0.580 | 0.052 | 1 | 0.603 0.643 0.548 0.597 0.509 |
| y3_recover_cash_6m | leftover od30 | 2,205 | 86 | 0.374 | 0.082 | -1 | 0.251 0.402 0.479 0.367 0.371 |
| y3_recover_cash_6m | clip24 leftover od30 | 2,205 | 86 | 0.621 | 0.066 | 1 | 0.637 0.538 0.636 0.713 0.580 |
| y3_recover_cash_6m | e_ar_overdue_30 | 2,655 | 143 | 0.629 | 0.137 | 1 | 0.806 0.537 0.543 0.747 0.511 |


## Extra — long-trail leftover fold-wise

Long so-far fold-wise leftover-iss 0.613 folds 0.755 0.618 0.700 0.579 0.415 n=828 pos=157 vs issued 0.646.

| feature | CV | folds | fold4 | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| DSO long | 0.607 | 0.758 0.615 0.655 0.593 0.416 | 0.416 | 828 | 157 |
| issued long | 0.646 | 0.631 0.601 0.552 0.742 0.705 | 0.705 | 916 | 209 |
| leftover iss long | 0.613 | 0.755 0.618 0.700 0.579 0.415 | 0.415 | 828 | 157 |
| days long | 0.408 | 0.512 0.262 0.506 0.473 0.285 | 0.285 | 916 | 209 |


## Extra — rolling-origin leftover (Y7 at t+3)

Rolling 9 origins (Y7 at t+3, leftover slope on < t): mean DSO 0.485 issued 0.596 leftover-iss 0.489. leftover dies on the origin path.

| t | pred | n | n_pos | DSO | issued | leftover |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-09-01 | 2025-12-01 | 408 | 133 | 0.509 | 0.579 | 0.522 |
| 2025-10-01 | 2026-01-01 | 432 | 144 | 0.528 | 0.577 | 0.531 |
| 2025-11-01 | 2026-02-01 | 439 | 132 | 0.527 | 0.641 | 0.529 |
| 2025-12-01 | 2026-03-01 | 453 | 105 | 0.409 | 0.632 | 0.410 |
| 2026-01-01 | 2026-04-01 | 531 | 130 | 0.517 | 0.513 | 0.518 |
| 2026-02-01 | 2026-05-01 | 555 | 126 | 0.419 | 0.634 | 0.425 |
| 2026-03-01 | 2026-06-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2026-04-01 | 2026-07-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2026-05-01 | 2026-08-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |


## Extra — rolling-origin leftover (Y3 at t+3)

Rolling Y3 at t+3 leftover-days: mean DSO — days — leftover —. leftover dies on the origin path.

| t | pred | n | n_pos | DSO | days | leftover |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-09-01 | 2025-12-01 | 415 | 32 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2025-10-01 | 2026-01-01 | 441 | 32 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2025-11-01 | 2026-02-01 | 445 | 34 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2025-12-01 | 2026-03-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2026-01-01 | 2026-04-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2026-02-01 | 2026-05-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2026-03-01 | 2026-06-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2026-04-01 | 2026-07-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2026-05-01 | 2026-08-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |


## Extra — |DSO|>24 persistence

|DSO|>24 persist: 213/673 companies at least once (31.6%), 126 twice (18.7%), 27 chronic-half. tail is mostly one-off, not a chronic object.

| slice | n | share |
| --- | --- | --- |
| companies with any DSO | 673 | 100.0% |
| |DSO|>24 at least once | 213 | 31.6% |
| |DSO|>24 at least twice | 126 | 18.7% |
| |DSO|>24 on ≥ half of nn months (nn≥4) | 27 | 4.0% |


## Extra — pooled Y3 t+3 leftover (per-origin LOW_POWER)

Pooled Y3 t+3 (origins=9 n=1301 pos=98): DSO 0.733 days 0.633 leftover 0.728 group-fold leftover —. pooled leftover lives — still do not KEEP.

| slice | n | n_pos | DSO | days | leftover |
| --- | --- | --- | --- | --- | --- |
| pooled Y3 t+3 | 1301 | 98 | 0.733 | 0.633 | 0.728 |
| pooled Y3 t+3 group-fold leftover | 1301 | 98 | — | — | — |


## Extra — rolling leftover after delay_coll (Y7 at t+3)

Rolling Y7 t+3 leftover-delay: mean DSO 0.485 delay 0.510 leftover 0.521. leftover dies on the origin path.

| t | pred | n | n_pos | DSO | delay | leftover |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-09-01 | 2025-12-01 | 408 | 133 | 0.509 | 0.528 | 0.523 |
| 2025-10-01 | 2026-01-01 | 432 | 144 | 0.528 | 0.517 | 0.510 |
| 2025-11-01 | 2026-02-01 | 439 | 132 | 0.527 | 0.525 | 0.530 |
| 2025-12-01 | 2026-03-01 | 453 | 105 | 0.409 | 0.452 | 0.472 |
| 2026-01-01 | 2026-04-01 | 531 | 130 | 0.517 | 0.512 | 0.545 |
| 2026-02-01 | 2026-05-01 | 555 | 126 | 0.419 | 0.527 | 0.544 |
| 2026-03-01 | 2026-06-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2026-04-01 | 2026-07-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |
| 2026-05-01 | 2026-08-01 | 0 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |


## Extra — leftover on |DSO|≤24 and fat issued

Clean |DSO|≤24 & issued>p10=3756: Y7 leftover-iss 0.400 issued 0.555 DSO 0.414. Y3 leftover-days 0.654 days 0.713. R²iss=0.000 R²days=0.000.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | Y7 leftover-iss clean | 5,550 | 1,213 | 0.400 | 0.069 | -1 | 0.422 0.433 0.408 0.454 0.281 |
| y7_top1_lost | Y7 issued clean | 5,550 | 1,213 | 0.555 | 0.070 | -1 | 0.545 0.516 0.487 0.555 0.671 |
| y7_top1_lost | Y7 DSO clean | 5,696 | 1,250 | 0.414 | 0.069 | -1 | 0.419 0.427 0.463 0.463 0.296 |
| y3_recover_cash_6m | Y3 leftover-days clean | 2,082 | 75 | 0.654 | 0.151 | -1 | 0.801 0.714 0.468 0.520 0.769 |
| y3_recover_cash_6m | Y3 days clean | 2,082 | 75 | 0.713 | 0.104 | -1 | 0.810 0.671 0.552 0.784 0.747 |
| y3_recover_cash_6m | Y3 DSO clean | 2,082 | 75 | 0.430 | 0.055 | 1 | 0.493 0.414 0.450 0.346 0.446 |


## Extra — clean-refit leftover (slope on the body only)

Clean-refit leftover: Y3 leftover-days 0.423 days 0.713 DSO 0.430 size 0.622 days+size leftover 0.433. Y7 leftover-iss 0.420 issued 0.555 DSO 0.414. R²days=0.000 R²iss=0.004 R²both=0.000. 0.654 was a full-slope artifact or still loses to days/size.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | Y7 leftover-iss refit | 5,550 | 1,213 | 0.420 | 0.071 | -1 | 0.413 0.433 0.481 0.470 0.302 |
| y7_top1_lost | Y7 issued clean | 5,550 | 1,213 | 0.555 | 0.070 | -1 | 0.545 0.516 0.487 0.555 0.671 |
| y7_top1_lost | Y7 DSO clean | 5,696 | 1,250 | 0.414 | 0.069 | -1 | 0.419 0.427 0.463 0.463 0.296 |
| y7_top1_lost | Y7 size clean | 5,369 | 1,167 | 0.437 | 0.084 | -1 | 0.459 0.464 0.443 0.523 0.297 |
| y3_recover_cash_6m | Y3 leftover-days refit | 2,082 | 75 | 0.423 | 0.049 | -1 | 0.480 0.412 0.445 0.347 0.432 |
| y3_recover_cash_6m | Y3 leftover days+size refit | 2,051 | 72 | 0.433 | 0.060 | -1 | 0.522 0.423 0.435 0.353 0.430 |
| y3_recover_cash_6m | Y3 days clean | 2,082 | 75 | 0.713 | 0.104 | -1 | 0.810 0.671 0.552 0.784 0.747 |
| y3_recover_cash_6m | Y3 DSO clean | 2,082 | 75 | 0.430 | 0.055 | 1 | 0.493 0.414 0.450 0.346 0.446 |
| y3_recover_cash_6m | Y3 size clean | 2,051 | 72 | 0.622 | 0.140 | -1 | 0.406 0.616 0.785 0.612 0.689 |


## Extra — pooled Y3 t+3 robust (tail / size / folds)

Pooled Y3 robust: DSO 0.733 days 0.633 size 0.584 leftover 0.728 drop>|24| 0.666 clip24 0.729 fold-pos [18, 9, 17, 35, 19]. not only a tail / still LOW_POWER folds — do not KEEP.

| slice | n | n_pos | AUROC |
| --- | --- | --- | --- |
| pooled Y3 t+3 DSO | 1301 | 98 | 0.733 |
| pooled Y3 t+3 days | 1301 | 98 | 0.633 |
| pooled Y3 t+3 size | 1301 | 98 | 0.584 |
| pooled leftover-days | 1301 | 98 | 0.728 |
| pooled leftover drop>|24| | 523 | 28 | 0.666 |
| pooled clip24 | 1301 | 98 | 0.729 |
| fold pos 0-4 | 1301 | 98 | 18 9 17 35 19 |


## Extra — |DSO|>24 by calendar month

|DSO|>24 by month: min 0.0% max 15.5% on 2026-08-01 (78/503). one month spike.

| period | nn | |DSO|>24 | share | p50 | p99 |
| --- | --- | --- | --- | --- | --- |
| 2024-09-01 | 169 | 0 | 0.0% | 0.97 | 1.0 |
| 2024-10-01 | 195 | 3 | 1.5% | 1.00 | 76.9 |
| 2024-11-01 | 200 | 6 | 3.0% | 1.03 | 207.7 |
| 2024-12-01 | 226 | 4 | 1.8% | 1.03 | 36.4 |
| 2025-01-01 | 253 | 9 | 3.6% | 1.22 | 182.3 |
| 2025-02-01 | 256 | 20 | 7.8% | 1.87 | 811.3 |
| 2025-03-01 | 276 | 22 | 8.0% | 1.73 | 505.6 |
| 2025-04-01 | 284 | 16 | 5.6% | 1.61 | 239.2 |
| 2025-05-01 | 298 | 20 | 6.7% | 1.67 | 1233.6 |
| 2025-06-01 | 317 | 17 | 5.4% | 1.50 | 1343.3 |
| 2025-07-01 | 330 | 19 | 5.8% | 1.59 | 574.1 |
| 2025-08-01 | 323 | 34 | 10.5% | 2.00 | 1593.8 |
| 2025-09-01 | 341 | 26 | 7.6% | 1.68 | 267.9 |
| 2025-10-01 | 354 | 21 | 5.9% | 1.69 | 1168.6 |
| 2025-11-01 | 361 | 26 | 7.2% | 2.13 | 884.6 |
| 2025-12-01 | 407 | 18 | 4.4% | 1.26 | 358.8 |
| 2026-01-01 | 447 | 44 | 9.8% | 2.09 | 630.2 |
| 2026-02-01 | 464 | 41 | 8.8% | 2.25 | 914.7 |
| 2026-03-01 | 502 | 40 | 8.0% | 1.79 | 274.9 |
| 2026-04-01 | 516 | 43 | 8.3% | 2.04 | 722.5 |
| 2026-05-01 | 510 | 39 | 7.6% | 2.00 | 808.0 |
| 2026-06-01 | 527 | 29 | 5.5% | 2.08 | 623.7 |
| 2026-07-01 | 524 | 41 | 7.8% | 2.10 | 419.6 |
| 2026-08-01 | 503 | 78 | 15.5% | 2.91 | 2170.4 |


## Extra — leftover after dropping 2026-08 spike

Drop 2026-08 spike: Y7 leftover-iss 0.452 leftover-delay 0.561 issued 0.630. Y3 leftover-days 0.582 days 0.711. Spike month is last grid month — labeled leftover barely moves.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | Y7 leftover-iss drop-Aug | 6,470 | 1,565 | 0.452 | 0.127 | 1 | 0.373 0.596 0.382 0.582 0.326 |
| y7_top1_lost | Y7 leftover-delay drop-Aug | 4,735 | 1,047 | 0.561 | 0.066 | -1 | 0.547 0.508 0.491 0.642 0.617 |
| y7_top1_lost | Y7 DSO drop-Aug | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 |
| y7_top1_lost | Y7 issued drop-Aug | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y3_recover_cash_6m | Y3 leftover-days drop-Aug | 2,418 | 93 | 0.582 | 0.126 | -1 | 0.495 0.721 0.462 0.516 0.715 |
| y3_recover_cash_6m | Y3 DSO drop-Aug | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 |
| y3_recover_cash_6m | Y3 days drop-Aug | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |


## Extra — Y6 leftover (not a new Y)

Y6 leftover (not a new Y): DSO 0.586 leftover-iss 0.584 leftover-days 0.610 issued 0.725 days 0.865 size 0.854. R²iss=0.000 R²days=0.000.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y6_zero_in_3 | Y6 DSO | 6,980 | 201 | 0.586 | 0.055 | 1 | 0.655 0.583 0.568 0.509 0.617 |
| y6_zero_in_3 | Y6 leftover-iss | 6,611 | 185 | 0.584 | 0.051 | 1 | 0.635 0.587 0.521 0.543 0.632 |
| y6_zero_in_3 | Y6 leftover-days | 6,980 | 201 | 0.610 | 0.047 | -1 | 0.582 0.665 0.580 0.657 0.567 |
| y6_zero_in_3 | Y6 issued_lag1 | 10,333 | 875 | 0.725 | 0.036 | -1 | 0.723 0.709 0.740 0.775 0.679 |
| y6_zero_in_3 | Y6 days | 17,083 | 1,490 | 0.865 | 0.017 | -1 | 0.842 0.884 0.852 0.871 0.874 |
| y6_zero_in_3 | Y6 size | 14,698 | 1,300 | 0.854 | 0.021 | -1 | 0.836 0.848 0.850 0.844 0.890 |
| y6_zero_in_3 | Y6 clip24 | 6,980 | 201 | 0.586 | 0.055 | 1 | 0.655 0.581 0.570 0.507 0.616 |


## Extra — leftover slope fit on labeled rows only

Labeled-slope leftover: Y3-days 0.697 (days 0.711) Y7-iss 0.452 (issued 0.630) Y7-delay 0.565. Y3 leftover lives on labeled slope / Y7 leftover dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | Y3 leftover-days labeled-slope | 2,418 | 93 | 0.697 | 0.095 | 1 | 0.706 0.701 0.544 0.804 0.733 |
| y3_recover_cash_6m | days labeled | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | Y3 leftover-size labeled-slope | 2,376 | 90 | 0.635 | 0.087 | 1 | 0.533 0.558 0.736 0.663 0.688 |
| y3_recover_cash_6m | size labeled | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y7_top1_lost | Y7 leftover-iss labeled-slope | 6,470 | 1,565 | 0.452 | 0.127 | 1 | 0.374 0.596 0.382 0.582 0.325 |
| y7_top1_lost | issued_lag1 labeled | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y7_top1_lost | Y7 leftover-delay labeled-slope | 4,735 | 1,047 | 0.565 | 0.064 | -1 | 0.551 0.518 0.493 0.642 0.619 |
| y7_top1_lost | delay labeled | 5,158 | 1,304 | 0.574 | 0.041 | 1 | 0.569 0.572 0.518 0.633 0.576 |
| y6_zero_in_3 | Y6 leftover-iss labeled-slope | 6,611 | 185 | 0.583 | 0.052 | 1 | 0.634 0.587 0.520 0.544 0.633 |
| y6_zero_in_3 | issued_lag1 labeled | 10,333 | 875 | 0.725 | 0.036 | -1 | 0.723 0.709 0.740 0.775 0.679 |


## Extra — rank(DSO) leftover (in-memory)

Rank(DSO) leftover: Y3 0.569 vs days 0.711 raw-rank 0.564. Y7 leftover-iss 0.431 leftover-delay 0.382 issued 0.630. R²days=0.000 R²iss=0.003 R²delay=0.073.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | Y3 rank(DSO) | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 |
| y3_recover_cash_6m | Y3 leftover rank-days | 2,418 | 93 | 0.569 | 0.134 | 1 | 0.684 0.383 0.584 0.701 0.491 |
| y3_recover_cash_6m | Y3 days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y7_top1_lost | Y7 rank(DSO) | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 |
| y7_top1_lost | Y7 leftover rank-iss | 6,470 | 1,565 | 0.431 | 0.101 | 1 | 0.364 0.599 0.423 0.427 0.341 |
| y7_top1_lost | Y7 leftover rank-delay | 4,735 | 1,047 | 0.382 | 0.057 | -1 | 0.400 0.340 0.428 0.438 0.306 |
| y7_top1_lost | Y7 issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |


## Extra — labeled-slope leftover vs days clone

Labeled leftover vs days ρ=-0.961 n=2418 (raw DSO-days -0.012). Leftover 0.697 days 0.711 clip-leftover 0.458. FALSE clone (tail rank).

| slice | n | value |
| --- | --- | --- |
| resid vs days ρ | 2418 | -0.961 |
| raw DSO vs days ρ (labeled) | 2418 | -0.012 |
| clip24-resid vs days ρ | 2418 | 0.120 |
| leftover AUROC | 2418 | 0.697 |
| days AUROC | 5648 | 0.711 |
| clip leftover AUROC | 2418 | 0.458 |
| R² labeled | 2418 | 0.000 |
| R² clip labeled | 2418 | 0.002 |


## Extra — DSO × log1p(issued) leftover

Mix DSO×log1p(iss) leftover-iss 0.448 vs DSO leftover 0.452 issued 0.630. Y3 mix leftover-days 0.692. R²dso=0.000 R²mix=0.000. Do not grow TURNOVER.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | Y7 leftover DSO after iss | 6,470 | 1,565 | 0.452 | 0.127 | 1 | 0.373 0.596 0.382 0.582 0.326 |
| y7_top1_lost | Y7 leftover mix after iss | 6,470 | 1,565 | 0.448 | 0.075 | -1 | 0.513 0.488 0.454 0.468 0.320 |
| y7_top1_lost | Y7 mix raw | 6,470 | 1,565 | 0.451 | 0.073 | -1 | 0.507 0.482 0.475 0.471 0.322 |
| y7_top1_lost | Y7 issued | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y3_recover_cash_6m | Y3 leftover DSO after days | 2,418 | 93 | 0.474 | 0.156 | -1 | 0.494 0.722 0.424 0.431 0.299 |
| y3_recover_cash_6m | Y3 leftover mix after days | 2,418 | 93 | 0.692 | 0.093 | 1 | 0.652 0.714 0.555 0.799 0.738 |
| y3_recover_cash_6m | Y3 days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |


## Extra — holdout DSO by month (coverage only)

Holdout DSO by month (coverage only, no AUROC): nn=450 / 1073 CM. Last-month spike is train-side 2026-08; holdout table is coverage.

| period | n_cm | nn | cov | |DSO|>24 | share |
| --- | --- | --- | --- | --- | --- |
| 2024-09-01 | 3 | 2 | 66.7% | 0 | 0.0% |
| 2024-10-01 | 3 | 3 | 100.0% | 0 | 0.0% |
| 2024-11-01 | 3 | 2 | 66.7% | 0 | 0.0% |
| 2024-12-01 | 4 | 3 | 75.0% | 0 | 0.0% |
| 2025-01-01 | 21 | 2 | 9.5% | 0 | 0.0% |
| 2025-02-01 | 21 | 2 | 9.5% | 0 | 0.0% |
| 2025-03-01 | 24 | 6 | 25.0% | 0 | 0.0% |
| 2025-04-01 | 27 | 8 | 29.6% | 1 | 12.5% |
| 2025-05-01 | 33 | 10 | 30.3% | 1 | 10.0% |
| 2025-06-01 | 43 | 17 | 39.5% | 0 | 0.0% |
| 2025-07-01 | 51 | 22 | 43.1% | 1 | 4.5% |
| 2025-08-01 | 51 | 22 | 43.1% | 1 | 4.5% |
| 2025-09-01 | 53 | 22 | 41.5% | 1 | 4.5% |
| 2025-10-01 | 55 | 24 | 43.6% | 1 | 4.2% |
| 2025-11-01 | 56 | 23 | 41.1% | 1 | 4.3% |
| 2025-12-01 | 59 | 26 | 44.1% | 0 | 0.0% |
| 2026-01-01 | 65 | 28 | 43.1% | 0 | 0.0% |
| 2026-02-01 | 71 | 31 | 43.7% | 1 | 3.2% |
| 2026-03-01 | 71 | 36 | 50.7% | 2 | 5.6% |
| 2026-04-01 | 71 | 34 | 47.9% | 2 | 5.9% |
| 2026-05-01 | 72 | 33 | 45.8% | 1 | 3.0% |
| 2026-06-01 | 72 | 34 | 47.2% | 1 | 2.9% |
| 2026-07-01 | 72 | 30 | 41.7% | 0 | 0.0% |
| 2026-08-01 | 72 | 30 | 41.7% | 3 | 10.0% |


## Extra — mix leftover vs days clone

Mix leftover vs days ρ=-0.983 (mix-days 0.084). Leftover 0.692 days 0.711. FALSE clone (tail rank).

| slice | n | value |
| --- | --- | --- |
| mix-resid vs days ρ | 8214 | -0.983 |
| mix vs days ρ | 8214 | 0.084 |
| mix leftover | 2418 | 0.692 |
| days | 5648 | 0.711 |
| R² | 8214 | 0.000 |


## Extra — leftover after pending / AP overdue

Leftover after pending: Y7 0.431 Y3 0.530 pending+days 0.530. After AP overdue Y7 0.449 Y3 0.547. R²pend=0.001 R²ap=0.000 R²pd=0.001.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | Y7 leftover pending | 6,651 | 1,623 | 0.431 | 0.058 | 1 | 0.444 0.437 0.485 0.456 0.332 |
| y7_top1_lost | Y7 leftover AP od | 6,510 | 1,588 | 0.449 | 0.102 | 1 | 0.594 0.443 0.499 0.370 0.339 |
| y7_top1_lost | Y7 pending | 7,464 | 2,149 | 0.420 | 0.040 | -1 | 0.410 0.410 0.467 0.450 0.363 |
| y3_recover_cash_6m | Y3 leftover pending | 2,418 | 93 | 0.530 | 0.064 | -1 | 0.573 0.551 0.593 0.503 0.433 |
| y3_recover_cash_6m | Y3 leftover pending+days | 2,418 | 93 | 0.530 | 0.064 | -1 | 0.573 0.549 0.593 0.502 0.432 |
| y3_recover_cash_6m | Y3 leftover AP od | 2,380 | 92 | 0.547 | 0.088 | -1 | 0.587 0.424 0.652 0.576 0.496 |
| y3_recover_cash_6m | Y3 days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | Y3 pending | 3,446 | 239 | 0.484 | 0.102 | 1 | 0.571 0.458 0.381 0.609 0.399 |


## Extra — leftover after delay_paid / zero_in / f_ds_r

More bars leftover: delay_paid Y7 0.471 Y3 0.376; zero_in Y7 0.392 Y3 0.574; f_ds_r Y7 0.424 Y3 0.566. R²paid=0.000 R²z=0.000 R²f=0.000.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | Y7 leftover delay_paid | 5,029 | 1,166 | 0.471 | 0.121 | 1 | 0.553 0.546 0.577 0.339 0.339 |
| y7_top1_lost | Y7 leftover zero_in | 6,651 | 1,623 | 0.392 | 0.034 | 1 | 0.378 0.386 0.423 0.427 0.347 |
| y7_top1_lost | Y7 leftover f_ds_r | 6,247 | 1,503 | 0.424 | 0.051 | -1 | 0.430 0.443 0.477 0.431 0.338 |
| y7_top1_lost | Y7 delay_paid | 5,591 | 1,522 | 0.444 | 0.096 | -1 | 0.518 0.438 0.562 0.342 0.358 |
| y3_recover_cash_6m | Y3 leftover delay_paid | 1,750 | 76 | 0.376 | 0.103 | 1 | 0.298 0.246 0.396 0.448 0.494 |
| y3_recover_cash_6m | Y3 leftover zero_in | 2,418 | 93 | 0.574 | 0.125 | 1 | 0.666 0.378 0.579 0.696 0.551 |
| y3_recover_cash_6m | Y3 leftover f_ds_r | 2,376 | 90 | 0.566 | 0.128 | -1 | 0.469 0.711 0.571 0.408 0.670 |
| y3_recover_cash_6m | Y3 leftover days (honest) | 2,418 | 93 | 0.474 | 0.156 | -1 | 0.494 0.722 0.424 0.431 0.299 |


## Extra — zero_in leftover vs days clone

Zero_in leftover vs days ρ=-0.058 vs zero_in 0.321. Leftover 0.574 days 0.711. not a days clone — still dies vs days / no KEEP.

| slice | n | value |
| --- | --- | --- |
| zero-resid vs days ρ | 8583 | -0.058 |
| zero-resid vs zero_in ρ | 8583 | 0.321 |
| leftover | 2418 | 0.574 |
| days | 5648 | 0.711 |
| R² | 8583 | 0.000 |


## Extra — leftover after zero_in+days

Leftover after zero_in+days 0.624 vs days 0.711 zero_in 0.580 R²=0.000. lives — still loses to days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover zero_in+days | 2,418 | 93 | 0.624 | 0.105 | -1 | 0.594 0.738 0.530 0.732 0.525 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | zero_in | 5,648 | 402 | 0.580 | 0.015 | 1 | 0.571 0.582 0.593 0.559 0.595 |


## Extra — zero_in+days leftover vs days clone

zero_in+days leftover vs days ρ=0.746. leftover 0.624. not a clone — still loses to days.

| slice | n | value |
| --- | --- | --- |
| resid vs days ρ | 8583 | 0.746 |
| leftover | 2418 | 0.624 |
| R² | 8583 | 0.000 |


## Extra — leftover after issued+days

Leftover after issued+days 0.691 vs days 0.711 issued 0.671 ρ=-0.974 R²=0.000. lives — still loses to days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover iss+days | 2,418 | 93 | 0.691 | 0.095 | 1 | 0.660 0.709 0.547 0.802 0.735 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | issued_lag1 | 3,618 | 264 | 0.671 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 |
|  |  | 8214 |  |  |  |  |  |


## Extra — clip24 leftover after issued+days

Clip24 leftover after issued+days 0.547 vs days 0.711 ρ=0.107 R²=0.003. 0.691 dies under clip.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | clip leftover iss+days | 2,418 | 93 | 0.547 | 0.135 | 1 | 0.640 0.364 0.595 0.685 0.449 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 8214 |  |  |  |  |  |


## Extra — leftover after f_ds_r+days

Leftover after f_ds_r+days 0.675 vs days 0.711 ρ=-0.937 R²=0.000. lives — still loses to days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover f_ds_r+days | 2,376 | 90 | 0.675 | 0.092 | 1 | 0.618 0.712 0.545 0.774 0.726 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 7813 |  |  |  |  |  |


## Extra — leftover after delay_paid+days

Leftover after delay_paid+days 0.624 vs days 0.711 ρ=0.702 R²=0.000. lives — still loses to days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover delay_paid+days | 1,750 | 76 | 0.624 | 0.074 | -1 | 0.533 0.642 0.582 0.732 0.631 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 6441 |  |  |  |  |  |


## Extra — leftover after AP overdue+days

Leftover after AP overdue+days 0.529 vs days 0.711 ρ=0.010 R²=0.000. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover AP+days | 2,380 | 92 | 0.529 | 0.098 | -1 | 0.569 0.407 0.655 0.558 0.458 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 8374 |  |  |  |  |  |


## Extra — leftover after DPO+days

Leftover after DPO+days 0.647 vs days 0.711 ρ=0.807 R²=0.003. lives — still loses to days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover DPO+days | 2,358 | 91 | 0.647 | 0.056 | -1 | 0.673 0.673 0.567 0.615 0.710 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 8236 |  |  |  |  |  |


## Extra — leftover after AR overdue+days

Leftover after AR overdue+days 0.541 vs days 0.711 ρ=0.121 R²=0.000. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover overdue+days | 2,205 | 86 | 0.541 | 0.104 | -1 | 0.644 0.563 0.535 0.593 0.371 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 8029 |  |  |  |  |  |


## Extra — clip leftover after delay_paid+days

Clip24 leftover after delay_paid+days 0.586 vs days 0.711 ρ=0.101 R²=0.030. lives — still loses to days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | clip leftover delay_paid+days | 1,750 | 76 | 0.586 | 0.093 | 1 | 0.697 0.515 0.574 0.662 0.480 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 6441 |  |  |  |  |  |


## Extra — clip leftover after DPO+days

Clip24 leftover after DPO+days 0.555 vs days 0.711 ρ=0.105 R²=0.005 raw DSO-days 0.022. not a days clone — still loses to days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | clip leftover DPO+days | 2,358 | 91 | 0.555 | 0.142 | 1 | 0.648 0.366 0.597 0.710 0.455 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 8236 |  |  |  |  |  |
|  |  | 8236 |  |  |  |  |  |


## Extra — leftover after AR open+days

Leftover after AR open+days 0.377 vs days 0.711 ρ=0.671 R²=0.000. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover open+days | 2,418 | 93 | 0.377 | 0.090 | -1 | 0.498 0.281 0.411 0.401 0.293 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 8583 |  |  |  |  |  |


## Extra — leftover after delay_coll+days

Leftover after delay_coll+days 0.647 vs days 0.711 ρ=0.878 R²=0.000. lives — still loses to days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover delay+days | 1,638 | 65 | 0.647 | 0.111 | -1 | 0.526 0.709 0.528 0.759 0.713 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 6091 |  |  |  |  |  |


## Extra — clip leftover after delay_coll+days

Clip24 leftover after delay_coll+days 0.608 vs days 0.711 ρ=0.027 R²=0.053. not a days clone — still loses to days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | clip leftover delay+days | 1,638 | 65 | 0.608 | 0.077 | 1 | 0.704 0.525 0.599 0.667 0.544 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 6091 |  |  |  |  |  |


## Extra — holdout SIZE tercile coverage

Holdout SIZE tercile coverage only: T1 33.9% T2 56.3% T3 37.7%. No AUROC.

| slice | n | nn | cov | |DSO|>24 |
| --- | --- | --- | --- | --- |
| holdout T1 | 310 | 105 | 33.9% | 1.0% |
| holdout T2 | 309 | 174 | 56.3% | 6.3% |
| holdout T3 | 310 | 117 | 37.7% | 1.7% |


## Extra — Y7 leftover after open+issued

Y7 leftover after open+issued 0.453 vs issued 0.630 ρ=0.219 R²=0.000. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover open+issued | 6,470 | 1,565 | 0.453 | 0.127 | 1 | 0.372 0.600 0.381 0.580 0.329 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 8214 |  |  |  |  |  |


## Extra — Y7 leftover after pending+issued

Y7 leftover after pending+issued 0.432 vs issued 0.630 ρ=0.031 R²=0.001. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover pending+issued | 6,470 | 1,565 | 0.432 | 0.069 | 1 | 0.452 0.437 0.498 0.457 0.314 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 8214 |  |  |  |  |  |


## Extra — Y7 leftover after AP+issued

Y7 leftover after AP+issued 0.449 vs issued 0.630 ρ=0.041 R²=0.001. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover AP+issued | 6,341 | 1,533 | 0.449 | 0.101 | 1 | 0.592 0.441 0.502 0.369 0.342 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 8044 |  |  |  |  |  |


## Extra — clip leftover after f_ds_r+days

Clip24 leftover after f_ds_r+days 0.547 vs days 0.711 ρ=0.117 R²=0.001. not a days clone — dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | clip leftover f_ds_r+days | 2,376 | 90 | 0.547 | 0.133 | 1 | 0.631 0.369 0.607 0.684 0.445 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 7813 |  |  |  |  |  |


## Extra — Y7 leftover after DPO+issued

Y7 leftover after DPO+issued 0.499 vs issued 0.630 ρ=0.150 R²=0.005. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover DPO+issued | 6,238 | 1,490 | 0.499 | 0.087 | 1 | 0.578 0.579 0.386 0.520 0.431 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 7914 |  |  |  |  |  |


## Extra — clip leftover after zero_in+days

Clip24 leftover after zero_in+days 0.560 vs days 0.711 ρ=0.083 R²=0.004. not a days clone — still loses to days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | clip leftover zero_in+days | 2,418 | 93 | 0.560 | 0.142 | 1 | 0.688 0.385 0.597 0.692 0.438 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 8583 |  |  |  |  |  |


## Extra — Y7 leftover after overdue+issued

Y7 leftover after overdue+issued 0.521 vs issued 0.630 ρ=0.079 R²=0.000. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover overdue+issued | 6,051 | 1,441 | 0.521 | 0.069 | -1 | 0.437 0.581 0.455 0.568 0.566 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 7699 |  |  |  |  |  |


## Extra — Y7 leftover after delay_paid+issued

Y7 leftover after delay_paid+issued 0.536 vs issued 0.630 ρ=-0.002 R²=0.001. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover delay_paid+issued | 4,964 | 1,157 | 0.536 | 0.120 | 1 | 0.543 0.539 0.599 0.341 0.659 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 6375 |  |  |  |  |  |


## Extra — Y7 leftover after size+issued

Y7 leftover after size+issued 0.463 vs issued 0.630 ρ=-0.512 R²=0.001. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover size+issued | 6,247 | 1,503 | 0.463 | 0.084 | 1 | 0.479 0.506 0.477 0.535 0.319 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 7813 |  |  |  |  |  |


## Extra — Y7 leftover after f_ds_r+issued

Y7 leftover after f_ds_r+issued 0.414 vs issued 0.630 ρ=0.220 R²=0.000. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover f_ds_r+issued | 6,247 | 1,503 | 0.414 | 0.047 | -1 | 0.436 0.443 0.435 0.425 0.331 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 7813 |  |  |  |  |  |


## Extra — leftover after AP issued+days

Leftover after AP issued+days 0.475 vs days 0.711 ρ=0.670 R²=0.000. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | leftover AP issued+days | 2,418 | 93 | 0.475 | 0.155 | -1 | 0.494 0.722 0.429 0.431 0.299 |
| y3_recover_cash_6m | days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
|  |  | 8583 |  |  |  |  |  |


## Extra — Y7 leftover after od30+issued

Y7 leftover after od30+issued 0.482 vs issued 0.630 ρ=-0.031 R²=0.000. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover od30+issued | 6,051 | 1,441 | 0.482 | 0.033 | 1 | 0.504 0.462 0.529 0.459 0.454 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 7699 |  |  |  |  |  |


## Extra — Y7 leftover after AP issued+issued

Y7 leftover after AP issued+issued 0.452 vs issued 0.630 ρ=0.231 R²=0.000. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover AP issued+issued | 6,470 | 1,565 | 0.452 | 0.128 | 1 | 0.373 0.598 0.382 0.582 0.326 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 8214 |  |  |  |  |  |


## Extra — Y7 leftover after zero_in+issued

Y7 leftover after zero_in+issued 0.506 vs issued 0.630 ρ=0.194 R²=0.000. dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | leftover zero_in+issued | 6,470 | 1,565 | 0.506 | 0.136 | 1 | 0.620 0.611 0.386 0.580 0.331 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
|  |  | 8214 |  |  |  |  |  |


## What failed / next (held for wave note)

- Y3 leftover after days 0.474 — CLOSE / DROP from the 44
- Y7 leftover dies after issued_lag1 0.452
- DROP e_dso_proxy from the 44
- Q6 CLOSE — exists early (unlike delay; early6 Y7 finite 87.3%) but Y7 lag1 short 0.422 dies — stock/flow is not a TURNOVER lead
- do not put DSO back on TURNOVER; night quote stays 0.720 / 0.712
- CONFIRM issued owns fold 4 (0.647 vs DSO 0.342)
- CONFIRM short-DSO Q1 hole univariate 0.455 (B_shallow OOF 0.410 locked; n=1331)
- Y3 leftover after days drop-tail 0.424 (raw 0.474) — not only a tail / still loses to days
- SHAP #1 not a delay twin; leftover-after-delay is a |DSO|>24 tail: leftover after delay 0.561 same-n 0.561; delay leftover KEEP 0.581
- Y3 Q6 lag1 0.588 short 0.602
- Y3 lag1 leftover-days 0.711 drop>24 0.553
- clip24 ICC 0.912 — do not write clip to store
- Y7 leftover-after-delay 0.561 is a |DSO|>24 tail (drop 0.424 clip24 0.404)
- Y3 leftover days+size 0.618 drop>24 0.418 (days-only leftover 0.474)
- Y3 lag1 leftover vs days ρ=-0.984 raw-lag1-days 0.023 FALSE clone (tail rank) leftover 0.711
- fat-issued leftover Y3 0.460 vs days 0.684
- chronic-85 Y3 DSO 0.564 → 0.566
- log1p leftover Y3-days 0.560 Y7-iss 0.431 Y7-delay 0.389
- DSO vs open ρ=0.600 Y7 open 0.465 open-leftover-iss 0.596
- lag1 vs days ρ=0.023 clip24-lag1 vs days 0.023 clip24-lag1 leftover 0.569
- Y3 leftover-after-issued 0.560 drop>24 0.416 tail
- max DSO 2190260.7 n>1000=70 n>1e5=5
- demean leftover-iss 0.461 mean leftover-iss 0.518
- Y3 fold leftover-days 0.474 fold4 0.299
- Q1 leftover-iss 0.472 leftover-delay 0.521 issued 0.604
- after7 leftover-delay 0.561 drop 0.424 leftover-iss 0.372
- long so-far DSO 0.607 leftover-iss 0.613 issued 0.646 short leftover-iss 0.503
- leftover after od30 Y7 0.477 clip 0.549 Y3 0.374
- long leftover fold-wise 0.613 n=828 pos=157 vs issued 0.646 — slice only
- rolling Y7 t+3 leftover-iss 0.489 DSO 0.485 issued 0.596
- rolling Y3 t+3 leftover-days — DSO — days —
- |DSO|>24 persist once 213 twice 126 chronic-half 27
- pooled Y3 t+3 leftover 0.728 days 0.633 n=1301 pos=98
- rolling Y7 t+3 leftover-delay 0.521 delay 0.510 DSO 0.485
- clean |DSO|≤24 leftover-iss 0.400 leftover-days 0.654 vs issued 0.555 days 0.713
- clean-refit leftover-days 0.423 DSO 0.430 days 0.713 size 0.622 leftover-iss 0.420
- pooled Y3 robust leftover 0.728 drop>|24| 0.666 DSO 0.733 days 0.633 size 0.584 not KEEP
- |DSO|>24 by month min 0.0% max 15.5% spike 2026-08-01
- drop-Aug leftover-iss 0.452 leftover-delay 0.561 leftover-days 0.582
- Y6 leftover-iss 0.584 DSO 0.586 issued 0.725 — not a new Y
- labeled-slope leftover Y3-days 0.697 Y7-iss 0.452 Y7-delay 0.565
- rank(DSO) leftover-days 0.569 leftover-iss 0.431 leftover-delay 0.382 raw-rank Y3 0.564
- labeled leftover vs days ρ=-0.961 raw -0.012 FALSE clone leftover 0.697 clip 0.458
- mix leftover-iss 0.448 DSO leftover 0.452 Y3 mix leftover-days 0.692
- holdout month coverage nn=450 (no AUROC)
- mix leftover vs days ρ=-0.983 leftover 0.692 FALSE clone
- leftover pending Y7 0.431 Y3 0.530 pending+days 0.530 AP Y7 0.449
- leftover delay_paid Y7 0.471 Y3 0.376 zero_in Y7 0.392 f_ds_r Y7 0.424
- zero_in leftover vs days ρ=-0.058 leftover 0.574 not a clone
- leftover zero_in+days 0.624 vs days 0.711
- zero_in+days leftover vs days ρ=0.746 leftover 0.624 not a clone
- leftover issued+days 0.691 vs days 0.711 ρ=-0.974
- clip leftover issued+days 0.547 ρ=0.107 vs days 0.711
- leftover f_ds_r+days 0.675 ρ=-0.937 vs days 0.711
- leftover delay_paid+days 0.624 ρ=0.702 vs days 0.711
- leftover AP+days 0.529 ρ=0.010 vs days 0.711
- leftover DPO+days 0.647 ρ=0.807 vs days 0.711
- leftover overdue+days 0.541 ρ=0.121 vs days 0.711
- clip leftover delay_paid+days 0.586 ρ=0.101 vs days 0.711
- clip leftover DPO+days 0.555 ρ=0.105 not a clone vs days 0.711
- leftover open+days 0.377 ρ=0.671 vs days 0.711
- leftover delay+days 0.647 ρ=0.878 vs days 0.711
- clip leftover delay+days 0.608 ρ=0.027 not a clone vs days 0.711
- holdout SIZE cov T1 33.9% T2 56.3% T3 37.7%
- leftover open+issued Y7 0.453 ρ=0.219 vs issued 0.630
- leftover pending+issued Y7 0.432 ρ=0.031 vs issued 0.630
- leftover AP+issued Y7 0.449 ρ=0.041 vs issued 0.630
- clip leftover f_ds_r+days 0.547 ρ=0.117 not a clone vs days 0.711
- leftover DPO+issued Y7 0.499 ρ=0.150 vs issued 0.630
- clip leftover zero_in+days 0.560 ρ=0.083 not a clone vs days 0.711
- leftover overdue+issued Y7 0.521 ρ=0.079 vs issued 0.630
- leftover delay_paid+issued Y7 0.536 ρ=-0.002 vs issued 0.630
- leftover size+issued Y7 0.463 ρ=-0.512 vs issued 0.630
- leftover f_ds_r+issued Y7 0.414 ρ=0.220 vs issued 0.630
- leftover AP issued+days 0.475 ρ=0.670 vs days 0.711
- leftover od30+issued Y7 0.482 ρ=-0.031 vs issued 0.630
- leftover AP issued+issued Y7 0.452 ρ=0.231 vs issued 0.630
- leftover zero_in+issued Y7 0.506 ρ=0.194 vs issued 0.630

Elapsed 13s. Cuts: coverage, SQL, twins, singles, leftover, |DSO|>24 tail, SIZE/book, Q6, ICC, fold 4 / short-Q1, delay leftover, holdout, same-n, quintiles, tiny-issued, Y3 same-n, fold4 groups, clip ICC, Y3 Q6, lag leftover, delay-tail, days+size, lag1-clone, fat issued, fold leftover, chronic, log1p, open, lag1-spearman, Y3-iss-tail, extreme rows, demean leftover, Y3 folds, Q1 leftover, after7, holdout tails, long leftover, od30, long folds, rolling Y7/Y3, persist, pool, delay-roll, clean/refit, pool-robust, month-tail, drop-Aug, Y6, labeled-slope, rank, labeled-clone, mix, holdout-month, mix-clone.
