# Q4/Q5 delay / overdue leftover after DSO / issued_lag1

Generated `2026-09-19T04:38:46+02:00` by agent `47815ca4`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_delay`. Night Y7 quote stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. Y7 never D. Y5 never E. Delay / overdue stay off the 15-col card. Y3 never B.

`e_delay_coll` / `e_delay_paid` = amount-weighted (paid − due) days on invoices paid in the trailing 3 months, clipped [−30, 120], **null the first 6 calendar months**. `e_ar_overdue` / `e_ap_overdue` = overdue |amount| / all-open unpaid. Javier overdue uses a 3-month issuance window.

## Headline

Delay vs DSO ρ=0.214 (not a twin). Y7 delay_coll 0.574 leftover after DSO 0.581 / issued_lag1 0.583 / both 0.584. Y3 0.512 leftover-days 0.427 vs days 0.711. 3m leftover 0.509. Q6 short 0.576. Y7 leftover **KEEP**. Y3 X **CLOSE / DROP from the 44**. Q6 **CLOSE**. PARK as health Y. Do not grow TURNOVER 0.720.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_delay`. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Not this clock. |
| 3 | Who is turning? | Delay lag is empty until month 7 — not a turn clock on short books. |
| 4 | Dip vs fall? | Y7 leftover after DSO / issued_lag1 **KEEP** — residual after DSO 0.581 and after issued_lag1 0.583 (both 0.584) survives and is not a twin (ρ DSO 0.214, issued_lag1 -0.053). Still do not grow TURNOVER or change 0.720. Leftover ≈ same-n raw — residualizing does not create a new object. ICC / demean: BETWEEN who-pays-late style, not a month shock. |
| 5 | Why did it change? | Delay on the SHAP card is Q5-shaped only if leftover lives. Fold 4: issued owns fold 4. 3m window leftover **CLOSE**. |
| 6 | Months earlier? | Delay empty first 6 calendar months — **CLOSE** as Q6 on short companies. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| delay_coll as Y7 leftover after DSO + issued_lag1 | **KEEP** | residual after DSO 0.581 and after issued_lag1 0.583 (both 0.584) survives and is not a twin (ρ DSO 0.214, issued_lag1 -0.053). Still do not grow TURNOVER or change 0.720. Leftover ≈ same-n raw — residualizing does not create a new object. ICC / demean: BETWEEN who-pays-late style, not a month shock. |
| delay / overdue as Y7 add-on / grow TURNOVER | **CLOSE** | do not grow TURNOVER; night quote stays 0.720 / 0.712 |
| delay / overdue as Y3 X / the 44 | **CLOSE / DROP from the 44** | Y3 delay_coll 0.512 loses to days 0.711 (leftover after days 0.427). |
| delay / overdue as a health Y | **PARK** | do not invent `y_delay` |
| Q6 delay lag1/lag3 on short books | **CLOSE** | Delay finite share on short so-far Y7 57.3%; on first-6 calendar Y7 0.0% (CONFIRM empty until month 7). Y7 delay lag1 short 0.576. CLOSE as Q6 — empty on the first 6 months / short books. |
| 3m-vs-all-open leftover | **CLOSE** | Y7 leftover of store AR after 3m-window 0.509 (raw store−3m 0.584, R²=0.744); AP after 3m 0.453. Leftover dies — the CLOSE 3m-vs-all-open window is not a health lever. |
| delay_paid as Y7 leftover | **CLOSE** | leftover after DSO 0.449; raw 0.444 |
| ar_overdue as Y7 leftover | **CLOSE** | leftover after DSO 0.547; raw 0.605 |
| Fold 4 Y7 | **issued owns fold 4** | delay 0.576 vs DSO 0.342 vs issued_lag1 0.647 vs TURNOVER 0.680 |

## 1. Coverage / nulls (first 6 empty; dark = NaN not 0)

Delay empty on first 6 calendar months (2024-09..2025-02): CONFIRM (coll 0.0% / paid 0.0%; DELAY_MASK_BEFORE=2025-03-01 ok). After month 7: coll 37.7% paid 48.0%. Dark never-ERP 470 (want 470): all six signals nn=0 / zero=0 (CONFIRM NaN not 0). Ever-ERP 744. Y7 labeled 7,464 pos 2,149; Y3 5,648 pos 402.

| col | nn | cov | early6 nn | after nn | dark nn / 0 | ERP nn | p50 | acf1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e_delay_coll | 6,752 | 31.9% | 0.0% | 37.7% | 0 / 0 | 49.8% | 2.58 | 0.626 |
| e_delay_paid | 8,602 | 40.7% | 0.0% | 48.0% | 0 / 0 | 63.5% | 1.09 | 0.600 |
| e_ar_overdue | 9,820 | 46.4% | 41.0% | 47.4% | 0 / 0 | 72.5% | 0.89 | 0.339 |
| e_ap_overdue | 12,173 | 57.5% | 54.6% | 58.1% | 0 / 0 | 89.8% | 0.68 | 0.508 |
| e_ar_overdue_30 | 9,820 | 46.4% | 41.0% | 47.4% | 0 / 0 | 72.5% | 0.58 | 0.606 |
| e_ap_overdue_30 | 12,173 | 57.5% | 54.6% | 58.1% | 0 / 0 | 89.8% | 0.40 | 0.618 |


Train CM 21,157 / companies 1,214. Y7 labeled 7,464 pos 2,149 rate 28.8%. Y3 stressed 5,648 pos 402 rate 7.1%.

Plot: `delay_vs_dso.png`.

## 2. Javier confirm (delay SAME; overdue 3m SAME / all-open CLOSE)

Delay vs Javier: coll 1.000 paid 1.000 (CONFIRM SAME ρ=1). Javier overdue vs in-module 3m: AR 1.000 AP 1.000 (CONFIRM ρ≈1). Store all-open vs 3m: AR 0.922 AP 0.886 (CONFIRM CLOSE). Javier overdue vs store all-open: AR 0.918 (CONFIRM 0.918) AP 0.883 (CONFIRM 0.883).

| pair | ρ | n | verdict |
| --- | --- | --- | --- |
| delay_coll | 1.000 | 6,752 | SAME |
| delay_paid | 1.000 | 8,602 | SAME |
| ar_overdue vs store | 0.918 | 8,178 | CLOSE |
| ap_overdue vs store | 0.883 | 10,615 | CLOSE |
| ar_overdue vs 3m | 1.000 | 8,178 | SAME |
| ap_overdue vs 3m | 1.000 | 10,615 | SAME |
| store AR vs 3m | 0.922 | 8,510 | CLOSE |
| store AP vs 3m | 0.886 | 11,070 | CLOSE |


## 3. Spearman twins (|ρ|≥0.80)

`e_delay_coll` vs DSO 0.214, issued_lag1 -0.053, ar_overdue_30 0.236, size -0.101, days -0.054. `e_ar_overdue` vs DSO 0.229. TWIN |ρ|≥0.80: e_ap_overdue_30 vs e_ap_overdue.

| pair | ρ | n | twin? |
| --- | --- | --- | --- |
| e_delay_coll vs e_dso_proxy | 0.214 | 6,091 |  |
| e_delay_coll vs e_ar_issued_lag1 | -0.053 | 6,681 |  |
| e_delay_coll vs e_ar_overdue_30 | 0.236 | 6,131 |  |
| e_delay_coll vs log1p(a_in3) | -0.101 | 6,589 |  |
| e_delay_coll vs c_n_days_with_tx | -0.054 | 6,752 |  |
| e_delay_coll vs e_ar_overdue | 0.278 | 6,131 |  |
| e_delay_coll vs e_ap_overdue | 0.285 | 6,617 |  |
| e_delay_coll vs e_delay_paid | 0.366 | 6,499 |  |
| e_delay_coll vs e_dpo_proxy | 0.147 | 6,453 |  |
| e_delay_paid vs e_dso_proxy | 0.216 | 6,441 |  |
| e_delay_paid vs e_ar_issued_lag1 | -0.008 | 8,495 |  |
| e_delay_paid vs e_ar_overdue_30 | 0.213 | 6,960 |  |
| e_delay_paid vs log1p(a_in3) | -0.045 | 8,338 |  |
| e_delay_paid vs c_n_days_with_tx | -0.017 | 8,602 |  |
| e_delay_paid vs e_ar_overdue | 0.181 | 6,960 |  |
| e_delay_paid vs e_ap_overdue | 0.329 | 8,334 |  |
| e_delay_paid vs e_dpo_proxy | 0.225 | 8,060 |  |
| e_ar_overdue vs e_dso_proxy | 0.229 | 8,029 |  |
| e_ar_overdue vs e_ar_issued_lag1 | -0.343 | 9,444 |  |
| e_ar_overdue vs e_ar_overdue_30 | 0.737 | 9,820 |  |
| e_ar_overdue vs log1p(a_in3) | -0.221 | 9,023 |  |
| e_ar_overdue vs c_n_days_with_tx | -0.163 | 9,820 |  |
| e_ar_overdue vs e_ap_overdue | 0.465 | 9,580 |  |
| e_ar_overdue vs e_delay_paid | 0.181 | 6,960 |  |
| e_ar_overdue vs e_dpo_proxy | 0.205 | 8,730 |  |
| e_ap_overdue vs e_dso_proxy | 0.288 | 8,374 |  |
| e_ap_overdue vs e_ar_issued_lag1 | -0.211 | 11,643 |  |
| e_ap_overdue vs e_ar_overdue_30 | 0.467 | 9,580 |  |
| e_ap_overdue vs log1p(a_in3) | -0.229 | 11,065 |  |
| e_ap_overdue vs c_n_days_with_tx | -0.183 | 12,173 |  |
| e_ap_overdue vs e_ar_overdue | 0.465 | 9,580 |  |
| e_ap_overdue vs e_delay_paid | 0.329 | 8,334 |  |
| e_ap_overdue vs e_dpo_proxy | 0.500 | 10,654 |  |
| e_ar_overdue_30 vs e_dso_proxy | 0.534 | 8,029 |  |
| e_ar_overdue_30 vs e_ar_issued_lag1 | -0.295 | 9,444 |  |
| e_ar_overdue_30 vs log1p(a_in3) | -0.139 | 9,023 |  |
| e_ar_overdue_30 vs c_n_days_with_tx | -0.112 | 9,820 |  |
| e_ar_overdue_30 vs e_ar_overdue | 0.737 | 9,820 |  |
| e_ar_overdue_30 vs e_ap_overdue | 0.467 | 9,580 |  |
| e_ar_overdue_30 vs e_delay_paid | 0.213 | 6,960 |  |
| e_ar_overdue_30 vs e_dpo_proxy | 0.369 | 8,730 |  |
| e_ap_overdue_30 vs e_dso_proxy | 0.383 | 8,374 |  |
| e_ap_overdue_30 vs e_ar_issued_lag1 | -0.155 | 11,643 |  |
| e_ap_overdue_30 vs e_ar_overdue_30 | 0.530 | 9,580 |  |
| e_ap_overdue_30 vs log1p(a_in3) | -0.173 | 11,065 |  |
| e_ap_overdue_30 vs c_n_days_with_tx | -0.128 | 12,173 |  |
| e_ap_overdue_30 vs e_ar_overdue | 0.427 | 9,580 |  |
| e_ap_overdue_30 vs e_ap_overdue | 0.849 | 12,173 | TWIN |
| e_ap_overdue_30 vs e_delay_paid | 0.289 | 8,334 |  |
| e_ap_overdue_30 vs e_dpo_proxy | 0.632 | 10,654 |  |


## 4. Single-feature train group-fold AUROC

Y7 delay_coll 0.574 paid 0.444 ar_od 0.605 ap_od 0.407 vs DSO 0.431 vs issued_lag1 0.630 (night 0.630, CONFIRM). Y3 delay_coll 0.512 vs days 0.711 (night 0.711, CONFIRM) vs size 0.617. Delay loses to days on Y3.

Sign from the train side of each fold. Seed 20260918. Night quotes: issued_lag1 0.630 (replica 0.630); days 0.711 (replica 0.711).

| y | feature | n | n_pos | CV | sd | sign | folds | present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_delay_coll | 5,158 | 1,304 | 0.574 | 0.041 | 1 | 0.569 0.572 0.518 0.633 0.576 | 69.1% |
| y7_top1_lost | e_delay_paid | 5,591 | 1,522 | 0.444 | 0.096 | -1 | 0.518 0.438 0.562 0.342 0.358 | 74.9% |
| y7_top1_lost | e_ar_overdue | 6,859 | 1,912 | 0.605 | 0.048 | 1 | 0.582 0.668 0.540 0.630 0.603 | 91.9% |
| y7_top1_lost | e_ap_overdue | 7,278 | 2,086 | 0.407 | 0.064 | 1 | 0.494 0.376 0.453 0.344 0.365 | 97.5% |
| y7_top1_lost | e_ar_overdue_30 | 6,859 | 1,912 | 0.580 | 0.052 | 1 | 0.603 0.643 0.548 0.597 0.509 | 91.9% |
| y7_top1_lost | e_ap_overdue_30 | 7,278 | 2,086 | 0.416 | 0.069 | -1 | 0.489 0.370 0.495 0.370 0.356 | 97.5% |
| y7_top1_lost | e_dso_proxy | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 | 89.1% |
| y7_top1_lost | e_ar_issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 | 97.2% |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 | 100.0% |
| y7_top1_lost | log1p_a_in3 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 | 93.8% |
| y7_top1_lost | ar_od_3m | 6,627 | 1,822 | 0.602 | 0.054 | 1 | 0.568 0.677 0.536 0.621 0.607 | 88.8% |
| y7_top1_lost | ap_od_3m | 7,121 | 2,019 | 0.422 | 0.069 | -1 | 0.516 0.399 0.469 0.358 0.367 | 95.4% |
| y3_recover_cash_6m | e_delay_coll | 1,819 | 86 | 0.512 | 0.087 | 1 | 0.426 0.455 0.626 0.582 0.473 | 32.2% |
| y3_recover_cash_6m | e_delay_paid | 2,294 | 150 | 0.521 | 0.098 | 1 | 0.419 0.441 0.543 0.665 0.536 | 40.6% |
| y3_recover_cash_6m | e_ar_overdue | 2,655 | 143 | 0.616 | 0.108 | 1 | 0.776 0.591 0.558 0.661 0.493 | 47.0% |
| y3_recover_cash_6m | e_ap_overdue | 3,336 | 223 | 0.625 | 0.126 | 1 | 0.772 0.561 0.520 0.750 0.520 | 59.1% |
| y3_recover_cash_6m | e_ar_overdue_30 | 2,655 | 143 | 0.629 | 0.137 | 1 | 0.806 0.537 0.543 0.747 0.511 | 47.0% |
| y3_recover_cash_6m | e_ap_overdue_30 | 3,336 | 223 | 0.620 | 0.142 | 1 | 0.787 0.548 0.491 0.759 0.515 | 59.1% |
| y3_recover_cash_6m | e_dso_proxy | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 | 42.8% |
| y3_recover_cash_6m | e_ar_issued_lag1 | 3,618 | 264 | 0.671 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 | 64.1% |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 | 100.0% |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 | 97.9% |
| y3_recover_cash_6m | ar_od_3m | 2,368 | 108 | 0.550 | 0.080 | 1 | 0.684 0.472 0.542 0.543 0.510 | 41.9% |
| y3_recover_cash_6m | ap_od_3m | 3,090 | 188 | 0.476 | 0.099 | 1 | 0.570 0.490 0.565 0.339 0.418 | 54.7% |


## 5. Residual Y7 after DSO and after issued_lag1

Y7 delay_coll leftover after DSO 0.581, after issued_lag1 0.583, after both 0.584. AR overdue leftover after DSO 0.547 after issued_lag1 0.587. Leftover survives ≥0.55 after DSO and issued_lag1.

KEEP leftover only if residual after DSO **and** after issued_lag1 survives (≥0.55) **and** not a twin. Still do **not** grow TURNOVER.

| stem | residual | n | n_pos | CV | R² | folds |
| --- | --- | --- | --- | --- | --- | --- |
| e_delay_coll | after DSO | 4,735 | 1,047 | 0.581 | 0.000 | 0.577 0.554 0.519 0.643 0.614 |
| e_delay_coll | after issued_lag1 | 5,088 | 1,291 | 0.583 | 0.002 | 0.582 0.573 0.513 0.643 0.606 |
| e_delay_coll | after DSO+issued_lag1 | 4,670 | 1,038 | 0.584 | 0.006 | 0.584 0.556 0.509 0.649 0.619 |
| e_delay_coll | after days | 5,158 | 1,304 | 0.566 | 0.013 | 0.586 0.587 0.525 0.614 0.517 |
| e_delay_coll | after size | 4,999 | 1,281 | 0.570 | 0.003 | 0.589 0.573 0.516 0.641 0.530 |
| e_delay_coll | after ar_overdue_30 | 4,688 | 1,119 | 0.523 | 0.060 | 0.512 0.468 0.498 0.569 0.566 |
| e_delay_paid | after DSO | 5,029 | 1,166 | 0.449 | 0.000 | 0.513 0.498 0.558 0.333 0.343 |
| e_delay_paid | after issued_lag1 | 5,519 | 1,506 | 0.446 | 0.004 | 0.510 0.434 0.573 0.342 0.369 |
| e_delay_paid | after DSO+issued_lag1 | 4,964 | 1,157 | 0.451 | 0.007 | 0.509 0.494 0.575 0.334 0.342 |
| e_delay_paid | after days | 5,591 | 1,522 | 0.441 | 0.013 | 0.536 0.410 0.561 0.360 0.339 |
| e_delay_paid | after size | 5,408 | 1,481 | 0.440 | 0.003 | 0.521 0.436 0.570 0.342 0.333 |
| e_delay_paid | after ar_overdue_30 | 5,114 | 1,339 | 0.535 | 0.045 | 0.575 0.470 0.571 0.414 0.646 |
| e_ar_overdue | after DSO | 6,216 | 1,490 | 0.547 | 0.000 | 0.482 0.604 0.487 0.582 0.578 |
| e_ar_overdue | after issued_lag1 | 6,666 | 1,845 | 0.587 | 0.000 | 0.555 0.641 0.530 0.619 0.588 |
| e_ar_overdue | after DSO+issued_lag1 | 6,051 | 1,441 | 0.554 | 0.000 | 0.488 0.618 0.486 0.591 0.586 |
| e_ar_overdue | after days | 6,859 | 1,912 | 0.604 | 0.009 | 0.598 0.695 0.539 0.602 0.583 |
| e_ar_overdue | after size | 6,427 | 1,768 | 0.592 | 0.021 | 0.593 0.665 0.533 0.608 0.561 |
| e_ar_overdue | after ar_overdue_30 | 6,859 | 1,912 | 0.537 | 0.590 | 0.511 0.524 0.469 0.573 0.610 |
| e_ap_overdue | after DSO | 6,510 | 1,588 | 0.428 | 0.000 | 0.553 0.409 0.477 0.355 0.347 |
| e_ap_overdue | after issued_lag1 | 7,084 | 2,015 | 0.407 | 0.001 | 0.490 0.379 0.455 0.343 0.370 |
| e_ap_overdue | after DSO+issued_lag1 | 6,341 | 1,533 | 0.429 | 0.002 | 0.553 0.413 0.484 0.355 0.339 |
| e_ap_overdue | after days | 7,278 | 2,086 | 0.413 | 0.016 | 0.519 0.368 0.448 0.364 0.366 |
| e_ap_overdue | after size | 6,848 | 1,935 | 0.405 | 0.033 | 0.502 0.390 0.450 0.350 0.334 |
| e_ap_overdue | after ar_overdue_30 | 6,716 | 1,864 | 0.474 | 0.204 | 0.494 0.571 0.525 0.381 0.397 |
| e_ar_overdue_30 | after DSO | 6,216 | 1,490 | 0.521 | 0.001 | 0.535 0.556 0.498 0.552 0.464 |
| e_ar_overdue_30 | after issued_lag1 | 6,666 | 1,845 | 0.574 | 0.000 | 0.595 0.637 0.542 0.597 0.499 |
| e_ar_overdue_30 | after DSO+issued_lag1 | 6,051 | 1,441 | 0.524 | 0.001 | 0.538 0.569 0.495 0.555 0.465 |
| e_ar_overdue_30 | after days | 6,859 | 1,912 | 0.578 | 0.012 | 0.609 0.654 0.550 0.581 0.495 |
| e_ar_overdue_30 | after size | 6,427 | 1,768 | 0.568 | 0.022 | 0.604 0.637 0.544 0.586 0.467 |
| e_ar_overdue_30 | after ar_overdue_30 | 6,859 | 1,912 | 0.561 | 1.000 | 0.585 0.621 0.531 0.583 0.484 |
| e_ap_overdue_30 | after DSO | 6,510 | 1,588 | 0.437 | 0.000 | 0.523 0.412 0.527 0.381 0.341 |
| e_ap_overdue_30 | after issued_lag1 | 7,084 | 2,015 | 0.418 | 0.000 | 0.497 0.372 0.494 0.368 0.362 |
| e_ap_overdue_30 | after DSO+issued_lag1 | 6,341 | 1,533 | 0.436 | 0.001 | 0.530 0.412 0.531 0.379 0.327 |
| e_ap_overdue_30 | after days | 7,278 | 2,086 | 0.413 | 0.021 | 0.468 0.367 0.485 0.379 0.369 |
| e_ap_overdue_30 | after size | 6,848 | 1,935 | 0.411 | 0.038 | 0.493 0.381 0.484 0.371 0.323 |
| e_ap_overdue_30 | after ar_overdue_30 | 6,716 | 1,864 | 0.440 | 0.259 | 0.492 0.419 0.492 0.395 0.402 |


## 6. Residual Y3 after days

Y3 delay_coll leftover after days 0.427 vs size 0.617 (Δ -0.190). Expect CLOSE as Y3 X — leftover after days dies.

| stem | residual | n | n_pos | CV | R² |
| --- | --- | --- | --- | --- | --- |
| e_delay_coll | after days | 1,819 | 86 | 0.427 | 0.013 |
| e_delay_coll | after size | 1,806 | 84 | 0.462 | 0.003 |
| e_delay_coll | after DSO | 1,638 | 65 | 0.417 | 0.000 |
| e_delay_coll | after days+size | 1,806 | 84 | 0.435 | 0.014 |
| e_delay_paid | after days | 2,294 | 150 | 0.519 | 0.013 |
| e_delay_paid | after size | 2,278 | 147 | 0.438 | 0.003 |
| e_delay_paid | after DSO | 1,750 | 76 | 0.378 | 0.000 |
| e_delay_paid | after days+size | 2,278 | 147 | 0.515 | 0.015 |
| e_ar_overdue | after days | 2,655 | 143 | 0.438 | 0.009 |
| e_ar_overdue | after size | 2,606 | 140 | 0.577 | 0.021 |
| e_ar_overdue | after DSO | 2,205 | 86 | 0.534 | 0.000 |
| e_ar_overdue | after days+size | 2,606 | 140 | 0.548 | 0.022 |
| e_ap_overdue | after days | 3,336 | 223 | 0.564 | 0.016 |
| e_ap_overdue | after size | 3,277 | 217 | 0.608 | 0.033 |
| e_ap_overdue | after DSO | 2,380 | 92 | 0.566 | 0.000 |
| e_ap_overdue | after days+size | 3,277 | 217 | 0.582 | 0.037 |
| e_ar_overdue_30 | after days | 2,655 | 143 | 0.580 | 0.012 |
| e_ar_overdue_30 | after size | 2,606 | 140 | 0.602 | 0.022 |
| e_ar_overdue_30 | after DSO | 2,205 | 86 | 0.372 | 0.001 |
| e_ar_overdue_30 | after days+size | 2,606 | 140 | 0.585 | 0.025 |
| e_ap_overdue_30 | after days | 3,336 | 223 | 0.576 | 0.021 |
| e_ap_overdue_30 | after size | 3,277 | 217 | 0.611 | 0.038 |
| e_ap_overdue_30 | after DSO | 2,380 | 92 | 0.547 | 0.000 |
| e_ap_overdue_30 | after days+size | 3,277 | 217 | 0.587 | 0.044 |


## 7. 3m-vs-all-open leftover

Y7 leftover of store AR after 3m-window 0.509 (raw store−3m 0.584, R²=0.744); AP after 3m 0.453. Leftover dies — the CLOSE 3m-vs-all-open window is not a health lever.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_ar_overdue all-open | 6,859 | 1,912 | 0.605 | 0.048 | 1 | 0.582 0.668 0.540 0.630 0.603 |
| y7_top1_lost | ar_od_3m | 6,627 | 1,822 | 0.602 | 0.054 | 1 | 0.568 0.677 0.536 0.621 0.607 |
| y7_top1_lost | store−3m AR | 6,627 | 1,822 | 0.584 | 0.034 | -1 | 0.545 0.631 0.560 0.578 0.605 |
| y7_top1_lost | AR resid after 3m | 6,627 | 1,822 | 0.509 | 0.034 | -1 | 0.472 0.474 0.526 0.521 0.550 |
| y7_top1_lost | e_ap_overdue all-open | 7,278 | 2,086 | 0.407 | 0.064 | 1 | 0.494 0.376 0.453 0.344 0.365 |
| y7_top1_lost | ap_od_3m | 7,121 | 2,019 | 0.422 | 0.069 | -1 | 0.516 0.399 0.469 0.358 0.367 |
| y7_top1_lost | store−3m AP | 7,121 | 2,019 | 0.551 | 0.061 | -1 | 0.514 0.494 0.556 0.539 0.651 |
| y7_top1_lost | AP resid after 3m | 7,121 | 2,019 | 0.453 | 0.072 | -1 | 0.516 0.430 0.513 0.466 0.340 |
| y3_recover_cash_6m | e_ar_overdue all-open | 2,655 | 143 | 0.616 | 0.108 | 1 | 0.776 0.591 0.558 0.661 0.493 |
| y3_recover_cash_6m | ar_od_3m | 2,368 | 108 | 0.550 | 0.080 | 1 | 0.684 0.472 0.542 0.543 0.510 |
| y3_recover_cash_6m | store−3m AR | 2,368 | 108 | 0.457 | 0.078 | -1 | 0.415 0.584 0.433 0.472 0.381 |
| y3_recover_cash_6m | AR resid after 3m | 2,368 | 108 | 0.539 | 0.140 | 1 | 0.709 0.416 0.590 0.604 0.375 |
| y3_recover_cash_6m | e_ap_overdue all-open | 3,336 | 223 | 0.625 | 0.126 | 1 | 0.772 0.561 0.520 0.750 0.520 |
| y3_recover_cash_6m | ap_od_3m | 3,090 | 188 | 0.476 | 0.099 | 1 | 0.570 0.490 0.565 0.339 0.418 |
| y3_recover_cash_6m | store−3m AP | 3,090 | 188 | 0.486 | 0.043 | -1 | 0.474 0.558 0.469 0.486 0.444 |
| y3_recover_cash_6m | AP resid after 3m | 3,090 | 188 | 0.555 | 0.091 | 1 | 0.648 0.487 0.575 0.627 0.437 |


## 8. Q6 — lag1/lag3 on short vs long

Delay finite share on short so-far Y7 57.3%; on first-6 calendar Y7 0.0% (CONFIRM empty until month 7). Y7 delay lag1 short 0.576. CLOSE as Q6 — empty on the first 6 months / short books.

| slice | col | n_nn | n_pos | present | CV |
| --- | --- | --- | --- | --- | --- |
| all | e_delay_coll | 5,158 | 1,304 | 69.1% | 0.574 |
| all | e_delay_coll_lag1 | 4,619 | 1,155 | 61.9% | 0.570 |
| all | e_delay_coll_lag3 | 3,631 | 916 | 48.6% | 0.555 |
| all | e_delay_paid | 5,591 | 1,522 | 74.9% | 0.444 |
| all | e_delay_paid_lag1 | 5,116 | 1,383 | 68.5% | 0.440 |
| all | e_ar_issued_lag1 | 7,253 | 2,072 | 97.2% | 0.630 |
| all | e_ar_overdue | 6,859 | 1,912 | 91.9% | 0.605 |
| all | e_ar_overdue_lag1 | 6,450 | 1,769 | 86.4% | 0.590 |
| short_<12_sofar | e_delay_coll | 2,535 | 641 | 57.3% | 0.581 |
| short_<12_sofar | e_delay_coll_lag1 | 2,056 | 507 | 46.5% | 0.576 |
| short_<12_sofar | e_delay_coll_lag3 | 1,179 | 295 | 26.7% | 0.572 |
| short_<12_sofar | e_delay_paid | 2,774 | 770 | 62.7% | 0.440 |
| short_<12_sofar | e_delay_paid_lag1 | 2,338 | 641 | 52.9% | 0.426 |
| short_<12_sofar | e_ar_issued_lag1 | 4,210 | 1,260 | 95.2% | 0.626 |
| short_<12_sofar | e_ar_overdue | 4,045 | 1,194 | 91.5% | 0.601 |
| short_<12_sofar | e_ar_overdue_lag1 | 3,677 | 1,061 | 83.2% | 0.584 |
| long_>=18_sofar | e_delay_coll | 806 | 168 | 88.0% | 0.561 |
| long_>=18_sofar | e_delay_coll_lag1 | 793 | 167 | 86.6% | 0.486 |
| long_>=18_sofar | e_delay_coll_lag3 | 774 | 173 | 84.5% | 0.485 |
| long_>=18_sofar | e_delay_paid | 864 | 201 | 94.3% | 0.548 |
| long_>=18_sofar | e_delay_paid_lag1 | 863 | 199 | 94.2% | 0.459 |
| long_>=18_sofar | e_ar_issued_lag1 | 916 | 209 | 100.0% | 0.646 |
| long_>=18_sofar | e_ar_overdue | 853 | 190 | 93.1% | 0.617 |
| long_>=18_sofar | e_ar_overdue_lag1 | 848 | 189 | 92.6% | 0.622 |
| short_<12_company | e_delay_coll | 477 | 59 | 61.6% | 0.567 |
| short_<12_company | e_delay_coll_lag1 | 356 | 39 | 46.0% | LOW_POWER |
| short_<12_company | e_delay_coll_lag3 | 162 | 18 | 20.9% | LOW_POWER |
| short_<12_company | e_delay_paid | 572 | 106 | 73.9% | 0.596 |
| short_<12_company | e_delay_paid_lag1 | 465 | 82 | 60.1% | 0.599 |
| short_<12_company | e_ar_issued_lag1 | 666 | 140 | 86.0% | 0.722 |
| short_<12_company | e_ar_overdue | 747 | 162 | 96.5% | 0.629 |
| short_<12_company | e_ar_overdue_lag1 | 592 | 119 | 76.5% | 0.650 |
| long_>=18_company | e_delay_coll | 4,349 | 1,164 | 69.8% | 0.573 |
| long_>=18_company | e_delay_coll_lag1 | 3,976 | 1,050 | 63.8% | 0.566 |
| long_>=18_company | e_delay_coll_lag3 | 3,265 | 847 | 52.4% | 0.551 |
| long_>=18_company | e_delay_paid | 4,674 | 1,326 | 75.0% | 0.443 |
| long_>=18_company | e_delay_paid_lag1 | 4,341 | 1,223 | 69.7% | 0.438 |
| long_>=18_company | e_ar_issued_lag1 | 6,154 | 1,813 | 98.8% | 0.616 |
| long_>=18_company | e_ar_overdue | 5,700 | 1,640 | 91.5% | 0.594 |
| long_>=18_company | e_ar_overdue_lag1 | 5,489 | 1,553 | 88.1% | 0.579 |
| early6_calendar | e_delay_coll | 0 | 0 | 0.0% | LOW_POWER |
| early6_calendar | e_delay_coll_lag1 | 0 | 0 | 0.0% | LOW_POWER |
| early6_calendar | e_delay_coll_lag3 | 0 | 0 | 0.0% | LOW_POWER |
| early6_calendar | e_delay_paid | 0 | 0 | 0.0% | LOW_POWER |
| early6_calendar | e_delay_paid_lag1 | 0 | 0 | 0.0% | LOW_POWER |
| early6_calendar | e_ar_issued_lag1 | 1,002 | 357 | 93.6% | 0.588 |
| early6_calendar | e_ar_overdue | 964 | 341 | 90.0% | 0.570 |
| early6_calendar | e_ar_overdue_lag1 | 861 | 288 | 80.4% | 0.435 |
| after_month7 | e_delay_coll | 5,158 | 1,304 | 80.7% | 0.574 |
| after_month7 | e_delay_coll_lag1 | 4,619 | 1,155 | 72.3% | 0.570 |
| after_month7 | e_delay_coll_lag3 | 3,631 | 916 | 56.8% | 0.555 |
| after_month7 | e_delay_paid | 5,591 | 1,522 | 87.5% | 0.444 |
| after_month7 | e_delay_paid_lag1 | 5,116 | 1,383 | 80.0% | 0.440 |
| after_month7 | e_ar_issued_lag1 | 6,251 | 1,715 | 97.8% | 0.643 |
| after_month7 | e_ar_overdue | 5,895 | 1,571 | 92.2% | 0.619 |
| after_month7 | e_ar_overdue_lag1 | 5,589 | 1,481 | 87.4% | 0.608 |


## 9. Fold 4 Y7 (DSO failed)

Fold 4 (sign from 0–3): delay_coll 0.576, DSO 0.342, issued_lag1 0.647, delay resid-both 0.619. TURNOVER fold-4 quote 0.680. TURNOVER fold-4 0.680 is issued, not delay.

| feature | CV | fold4 | folds |
| --- | --- | --- | --- |
| e_delay_coll | 0.574 | 0.576 | 0.569 0.572 0.518 0.633 0.576 |
| e_delay_paid | 0.444 | 0.358 | 0.518 0.438 0.562 0.342 0.358 |
| e_ar_overdue | 0.605 | 0.603 | 0.582 0.668 0.540 0.630 0.603 |
| e_ar_overdue_30 | 0.580 | 0.509 | 0.603 0.643 0.548 0.597 0.509 |
| e_dso_proxy | 0.431 | 0.342 | 0.370 0.600 0.420 0.424 0.342 |
| e_ar_issued_lag1 | 0.630 | 0.647 | 0.643 0.662 0.590 0.605 0.647 |
| delay resid after DSO | 0.581 | 0.614 | 0.577 0.554 0.519 0.643 0.614 |
| delay resid after issued_lag1 | 0.583 | 0.606 | 0.582 0.573 0.513 0.643 0.606 |
| delay resid after both | 0.584 | 0.619 | 0.584 0.556 0.509 0.649 0.619 |


Fold-4-only (sign from folds 0–3):

| feature | n_va | n_pos | fold4 | sign |
| --- | --- | --- | --- | --- |
| e_delay_coll | 1366 | 509 | 0.576 | 1 |
| e_delay_paid | 1530 | 579 | 0.358 | 1 |
| e_ar_overdue | 1473 | 494 | 0.603 | 1 |
| e_ar_overdue_30 | 1473 | 494 | 0.509 | 1 |
| e_dso_proxy | 1553 | 528 | 0.342 | 1 |
| e_ar_issued_lag1 | 1734 | 657 | 0.647 | -1 |
| delay resid after DSO | 1219 | 419 | 0.614 | 1 |
| delay resid after issued_lag1 | 1353 | 506 | 0.606 | 1 |
| delay resid after both | 1206 | 416 | 0.619 | 1 |


Problem groups:

| group | n_lab | n_pos | rate | delay p50 | DSO p50 |
| --- | --- | --- | --- | --- | --- |
| GROUP_0222 | 336 | 241 | 71.7% | 2.88 | 0.27 |
| GROUP_0108 | 204 | 130 | 63.7% | 0.00 | 0.00 |


## 10. ICC / company-demean

delay_coll ICC=0.922 k=594; DSO ICC=0.628. Y7 demean 0.521 company-mean 0.569. Company-mean carries the skill — style / truncation dummy.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_delay_coll raw | 5,158 | 1,304 | 0.574 | 0.041 | 1 | 0.569 0.572 0.518 0.633 0.576 |
| y7_top1_lost | e_delay_coll demean | 5,158 | 1,304 | 0.521 | 0.034 | 1 | 0.552 0.512 0.501 0.481 0.562 |
| y7_top1_lost | e_delay_coll company-mean | 6,900 | 1,973 | 0.569 | 0.088 | 1 | 0.497 0.635 0.530 0.689 0.495 |
| y3_recover_cash_6m | e_delay_coll raw | 1,819 | 86 | 0.512 | 0.087 | 1 | 0.426 0.455 0.626 0.582 0.473 |
| y3_recover_cash_6m | e_delay_coll demean | 1,819 | 86 | 0.418 | 0.059 | -1 | 0.409 0.501 0.408 0.337 0.438 |
| y3_recover_cash_6m | e_delay_coll company-mean | 3,017 | 168 | 0.576 | 0.048 | 1 | 0.565 0.578 0.551 0.654 0.530 |
| y7_top1_lost | e_delay_paid raw | 5,591 | 1,522 | 0.444 | 0.096 | -1 | 0.518 0.438 0.562 0.342 0.358 |
| y7_top1_lost | e_delay_paid demean | 5,591 | 1,522 | 0.478 | 0.032 | -1 | 0.504 0.492 0.497 0.424 0.475 |
| y7_top1_lost | e_delay_paid company-mean | 7,072 | 2,038 | 0.426 | 0.145 | -1 | 0.571 0.362 0.591 0.277 0.331 |
| y3_recover_cash_6m | e_delay_paid raw | 2,294 | 150 | 0.521 | 0.098 | 1 | 0.419 0.441 0.543 0.665 0.536 |
| y3_recover_cash_6m | e_delay_paid demean | 2,294 | 150 | 0.460 | 0.061 | 1 | 0.399 0.440 0.475 0.558 0.427 |
| y3_recover_cash_6m | e_delay_paid company-mean | 3,406 | 231 | 0.550 | 0.094 | 1 | 0.589 0.462 0.597 0.661 0.440 |
| y7_top1_lost | e_ar_overdue raw | 6,859 | 1,912 | 0.605 | 0.048 | 1 | 0.582 0.668 0.540 0.630 0.603 |
| y7_top1_lost | e_ar_overdue demean | 6,859 | 1,912 | 0.534 | 0.020 | 1 | 0.548 0.509 0.532 0.524 0.560 |
| y7_top1_lost | e_ar_overdue company-mean | 7,386 | 2,109 | 0.581 | 0.074 | 1 | 0.551 0.688 0.510 0.626 0.532 |
| y3_recover_cash_6m | e_ar_overdue raw | 2,655 | 143 | 0.616 | 0.108 | 1 | 0.776 0.591 0.558 0.661 0.493 |
| y3_recover_cash_6m | e_ar_overdue demean | 2,655 | 143 | 0.569 | 0.133 | 1 | 0.691 0.465 0.524 0.730 0.437 |
| y3_recover_cash_6m | e_ar_overdue company-mean | 3,346 | 201 | 0.565 | 0.096 | 1 | 0.709 0.561 0.572 0.540 0.441 |
| y7_top1_lost | e_dso_proxy raw | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 |
| y7_top1_lost | e_dso_proxy demean | 6,651 | 1,623 | 0.475 | 0.035 | 1 | 0.430 0.489 0.523 0.479 0.453 |
| y7_top1_lost | e_dso_proxy company-mean | 7,462 | 2,147 | 0.489 | 0.133 | 1 | 0.341 0.620 0.558 0.576 0.352 |
| y3_recover_cash_6m | e_dso_proxy raw | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 |
| y3_recover_cash_6m | e_dso_proxy demean | 2,418 | 93 | 0.541 | 0.077 | 1 | 0.610 0.601 0.418 0.544 0.532 |
| y3_recover_cash_6m | e_dso_proxy company-mean | 3,398 | 205 | 0.396 | 0.038 | -1 | 0.416 0.349 0.446 0.394 0.373 |


## 11. Holdout coverage only (no AUROC)

Holdout 72 coverage only: 1,073 CM / 72 companies. Y7 pos=122 (quote 122). delay_coll cov 40.7%; early6 0.0%; dark nn=0. No AUROC claim.

| col | n_cm | nn | cov | early6 nn | dark nn |
| --- | --- | --- | --- | --- | --- |
| e_delay_coll | 1,073 | 437 | 40.7% | 0.0% | 0 |
| e_delay_paid | 1,073 | 466 | 43.4% | 0.0% | 0 |
| e_ar_overdue | 1,073 | 477 | 44.5% | 20.0% | 0 |
| e_ap_overdue | 1,073 | 528 | 49.2% | 23.6% | 0 |
| e_ar_overdue_30 | 1,073 | 477 | 44.5% | 20.0% | 0 |
| e_ap_overdue_30 | 1,073 | 528 | 49.2% | 23.6% | 0 |


## Extra 12 — first-6 NaNs vs short books

Companies never finite delay_coll: 620 / 1214 (of which short-book 208). Calendar mask (not company age) zeros the first 6 months for everyone.

| slice | n_cm | nn | cov | p50 |
| --- | --- | --- | --- | --- |
| calendar first-6 | 3229 | 0 | 0.0% | — |
| calendar after-6 | 17928 | 6752 | 37.7% | 2.58 |
| so_far <7 | 7279 | 968 | 13.3% | 0.00 |
| so_far 7-11 | 5195 | 2036 | 39.2% | 2.00 |
| so_far >=12 | 8683 | 3748 | 43.2% | 4.37 |
| company short_<12 | 2970 | 872 | 29.4% | 0.00 |
| company long_>=18 | 16081 | 5428 | 33.8% | 3.76 |


## Extra 13 — short-DSO fifth

Short-DSO fifth (Q1): delay CV 0.516 vs DSO 0.455. Same short-DSO hole — delay does not save Q1.

| DSO q | n | rate | DSO p50 | delay nn | delay CV | DSO CV | issued CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | 1331 | 25.5% | 0.06 | 80.5% | 0.516 | 0.455 | 0.604 |
| Q2 | 1330 | 25.0% | 1.00 | 68.2% | 0.601 | 0.449 | 0.647 |
| Q3 | 1330 | 22.1% | 1.74 | 75.8% | 0.604 | 0.582 | 0.448 |
| Q4 | 1330 | 19.8% | 3.74 | 70.6% | 0.536 | 0.546 | 0.521 |
| Q5 | 1330 | 29.6% | 15.45 | 60.8% | 0.527 | 0.644 | 0.644 |


## Extra 14 — delay vs overdue_30

delay_coll vs ar_overdue_30 ρ=0.236 (not a twin). Y7 leftover after overdue_30 0.523 R²=0.060.

## Extra 15 — chronic 12 (Y3 only)

Chronic 12 Y2 names: 12. Y3 delay 0.512 → drop-12 0.512 (does not flip). No Y2 AUROC claim.

| slice | delay | days |
| --- | --- | --- |
| Y3 all | 0.512 | 0.711 |
| Y3 drop-12 | 0.512 | 0.708 |


## Extra 16 — quintile rates

Y7 / Y3 rates by delay_coll quintile (train labeled, delay finite).

| y | delay q | n | n_pos | rate | delay p50 |
| --- | --- | --- | --- | --- | --- |
| y7_top1_lost | Q1 | 1032 | 193 | 18.7% | -1.73 |
| y7_top1_lost | Q2 | 1031 | 244 | 23.7% | 0.00 |
| y7_top1_lost | Q3 | 1032 | 319 | 30.9% | 2.22 |
| y7_top1_lost | Q4 | 1031 | 269 | 26.1% | 12.00 |
| y7_top1_lost | Q5 | 1032 | 279 | 27.0% | 43.96 |
| y3_recover_cash_6m | Q1 | 364 | 12 | 3.3% | -3.07 |
| y3_recover_cash_6m | Q2 | 364 | 25 | 6.9% | 0.00 |
| y3_recover_cash_6m | Q3 | 363 | 11 | 3.0% | 1.60 |
| y3_recover_cash_6m | Q4 | 364 | 17 | 4.7% | 11.84 |
| y3_recover_cash_6m | Q5 | 364 | 21 | 5.8% | 41.89 |


## Extra 17 — leftover slices

Y7 delay raw vs leftover-after-DSO+issued_lag1 inside ERP / post-truncation slices.

| slice | raw | leftover both | n | n_pos |
| --- | --- | --- | --- | --- |
| all labeled | 0.574 | 0.584 | 5,158 | 1,304 |
| after month 7 | 0.574 | 0.584 | 5,158 | 1,304 |
| ever ERP | 0.574 | 0.584 | 5,158 | 1,304 |
| delay finite | 0.574 | 0.584 | 5,158 | 1,304 |


## Extra 18 — clip pile

delay_coll clip pile: lo 89 hi 241 / 6,752 defined (clip is a tail).

| edge | n | share_nn |
| --- | --- | --- |
| clip lo -30 | 89 | 1.3% |
| clip hi 120 | 241 | 3.6% |
| interior | 6422 | 95.1% |


## Extra 19 — same-n raw vs leftover

Same-n (delay+DSO+issued_lag1 finite): raw delay 0.579 leftover 0.584 issued 0.578 DSO 0.371. Leftover ≈ raw — residualizing DSO/issued does not create a new object.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | delay same-n | 4,670 | 1,038 | 0.579 | 0.047 | 1 | 0.581 0.558 0.518 0.644 0.595 |
| y7_top1_lost | issued_lag1 same-n | 4,670 | 1,038 | 0.578 | 0.064 | -1 | 0.601 0.523 0.506 0.599 0.662 |
| y7_top1_lost | DSO same-n | 4,670 | 1,038 | 0.371 | 0.050 | -1 | 0.377 0.327 0.430 0.406 0.314 |
| y7_top1_lost | leftover both same-n | 4,670 | 1,038 | 0.584 | 0.054 | 1 | 0.584 0.556 0.509 0.649 0.619 |


## Extra 20 — company-mean vs demean leftover

Y7 company-mean delay 0.569 demean 0.521; mean|DSO 0.570 demean|DSO 0.492. BETWEEN who-pays-late style — not a month dip-vs-fall shock.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | company-mean delay | 6,900 | 1,973 | 0.569 | 0.088 | 1 | 0.497 0.635 0.530 0.689 0.495 |
| y7_top1_lost | demean delay | 5,158 | 1,304 | 0.521 | 0.034 | 1 | 0.552 0.512 0.501 0.481 0.562 |
| y7_top1_lost | mean after DSO | 6,200 | 1,528 | 0.570 | 0.089 | 1 | 0.490 0.623 0.557 0.694 0.489 |
| y7_top1_lost | demean after DSO | 4,735 | 1,047 | 0.492 | 0.048 | 1 | 0.566 0.499 0.485 0.477 0.434 |


## Extra 21 — fold-wise leftover

Y7 fold-wise delay vs leftover vs issued_lag1 / DSO.

| feature | CV | folds | fold4 |
| --- | --- | --- | --- |
| e_delay_coll | 0.574 | 0.569 0.572 0.518 0.633 0.576 | 0.576 |
| e_ar_issued_lag1 | 0.630 | 0.643 0.662 0.590 0.605 0.647 | 0.647 |
| e_dso_proxy | 0.431 | 0.370 0.600 0.420 0.424 0.342 | 0.342 |
| leftover both | 0.584 | 0.584 0.556 0.509 0.649 0.619 | 0.619 |


## Extra 22 — Q6 after month 7 only

After month 7 only: delay now 0.574 lag1 0.570 vs issued_lag1 0.643. CLOSE as Q6 — lag1 does not hold issued_lag1 0.630 even after the mask.

| y | feature | n | n_pos | CV | sd | sign | folds | present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_delay_coll | 5,158 | 1,304 | 0.574 | 0.041 | 1 | 0.569 0.572 0.518 0.633 0.576 | 80.7% |
| y7_top1_lost | e_delay_coll_lag1 | 4,619 | 1,155 | 0.570 | 0.037 | 1 | 0.533 0.574 0.537 0.621 0.586 | 72.3% |
| y7_top1_lost | e_delay_coll_lag3 | 3,631 | 916 | 0.555 | 0.041 | 1 | 0.520 0.542 0.517 0.608 0.589 | 56.8% |
| y7_top1_lost | e_ar_issued_lag1 | 6,251 | 1,715 | 0.643 | 0.026 | -1 | 0.652 0.648 0.599 0.670 0.646 | 97.8% |
| y7_top1_lost | e_ar_overdue | 5,895 | 1,571 | 0.619 | 0.051 | 1 | 0.601 0.650 0.546 0.681 0.616 | 92.2% |
| y7_top1_lost | e_ar_overdue_lag1 | 5,589 | 1,481 | 0.608 | 0.046 | 1 | 0.596 0.626 0.546 0.673 0.600 | 87.4% |


## Extra 23 — who-is-late terciles

Company-mean delay terciles: Y7 rate spread 6.8%. A BETWEEN late-payer style would show a monotone Y7 gap.

| y | tercile | n_cm | n_co | n_lab | n_pos | rate |
| --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | late-low | 3305 | 198 | 1804 | 446 | 24.7% |
| y7_top1_lost | late-mid | 3633 | 198 | 2349 | 663 | 28.2% |
| y7_top1_lost | late-high | 4080 | 198 | 2747 | 864 | 31.5% |
| y3_recover_cash_6m | late-low | 3305 | 198 | 927 | 58 | 6.3% |
| y3_recover_cash_6m | late-mid | 3633 | 198 | 1026 | 35 | 3.4% |
| y3_recover_cash_6m | late-high | 4080 | 198 | 1064 | 75 | 7.0% |


## Extra 24 — overdue as Y3 leftover

Y3 ar_overdue 0.616 vs days 0.711 vs size 0.617; leftover after days 0.438. CLOSE / DROP from the 44.

## Extra 25 — delay_paid / overdue Y7 split

Y7 delay_paid leftover after DSO 0.449 (CLOSE); AR overdue leftover after DSO 0.547 (CLOSE). Raw delay_paid 0.444 ar_od 0.605.

## Extra 26 — leftover vs issued same-n / late joiners

Same-n leftover − issued_lag1 Δ=0.005 (leftover does not beat issued on the overlap). Late-joiner so_far<7 after calendar month 7: delay cov 23.9% (calendar mask ≠ company age).

## What failed / next (held for wave note)

- Y3 leftover after days 0.427 — CLOSE as Y3 X
- CLOSE as Q6 — delay empty until month 7
- 3m-vs-all-open leftover dies — CLOSE window is not a health lever
- fold 4 delay 0.576 does not save the short-DSO hole
- same-n leftover 0.584 ≈ raw 0.579 — not a new object
- BETWEEN style: company-mean 0.569 demean 0.521 ICC=0.922
- delay_paid leftover dies 0.449 — CLOSE
- AR overdue leftover after DSO dies 0.547 — CLOSE
- same-n leftover − issued Δ=0.005 — do not grow TURNOVER

Elapsed 9s. Cuts: coverage, Javier 3m, twins, singles, Y7 leftover after DSO/issued_lag1, Y3 leftover after days, 3m window leftover, Q6 short, fold 4, ICC, holdout, first-6 vs short books, short-DSO fifth, overdue_30, chronic-12, quintiles, leftover slices, clip, same-n leftover, company-mean vs demean, fold-wise leftover, Q6 after-month-7, who-is-late, overdue Y3 leftover, delay_paid/overdue split, leftover vs issued same-n.
