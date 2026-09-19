# Q4/Q5/Q6 unused `e_pending_amt_share` leftover after DSO / issued

Generated `2026-09-19T04:42:53+02:00` by agent `86399952`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_pending`. Night Y7 quote stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. Y7 never D. Y5 never E. Pending stays off the 15-col Y3 card. Do not grow TURNOVER.

`e_pending_amt_share` = among issued non-cancel invoices with issuance_date ≤ period_end, abs-amount share still unpaid as of period_end (invalid `payment_date` treated as 0 in both num and den). 470 never-invoice companies stay **NaN not 0**. Snapshot `pending_amount` is as-of extraction and would leak later collections — not used.

## Headline

**DROP** as Y7 X (DROP). Y3 X **DROP**. Leftover Y7 after DSO 0.421 / issued_lag1 0.418 / both 0.419; Y3 after days 0.443. not SIZE (ρ=0.063); not a |ρ|≥0.80 twin. Q6 CLOSE: lag1 0.430 now 0.420; first-6m pending 58.3% vs delay 0.0%. U-shape |p−0.30| leftover after issued 0.558 is a thin dummy (CLOSE). 44 should lose `e_pending_amt_share`. Night quotes unchanged: TURNOVER 0.720 / B_shallow 0.712; Y3 0.762 / 0.752; days 0.711; size 0.617.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_pending`. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Not this stock share. |
| 3 | Who is turning? | Pending lag is persistence (trait), not a turn clock. |
| 4 | Dip vs fall? | **DROP** — leftover after DSO 0.421 and issued_lag1 0.418 both <0.55. Unused stock. TURNPEND 0.7184 vs TURNOVER 0.72 already lost. |
| 5 | Why did it change? | **PARK** — ICC 0.971 η² 0.672 — who-has-unpaid-stock trait (like a_out_vol η²=0.74), not this-month shock. Demean Y7 0.541. |
| 6 | Months earlier? | **CLOSE** — now 0.420 lag1 0.430 lag3 0.443 — persistence, not lead. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| pending as Y7 X / leftover after DSO+issued_lag1 | DROP | leftover after DSO 0.421 and issued_lag1 0.418 both <0.55. Unused stock. TURNPEND 0.7184 vs TURNOVER 0.72 already lost. |
| pending as TURNOVER add-on (TURNPEND) | CLOSE | TURNPEND 0.7184 vs TURNOVER 0.72 — already lost. Do not grow TURNOVER. |
| pending as Y3 X / the 15-col card | DROP | Y3 leftover after days 0.443 dies vs days bar 0.711. Do not put pending on the 15-col Y3 card. |
| e_pending_amt_share on the 44-col keep list | DROP from the 44 | cluster representative in the feature report, but leftover dies and TURNPEND lost. BETWEEN trait, not a usable X. |
| pending as a health Y | PARK | do not invent y_pending. |
| Q5 why (unpaid stock) | PARK | ICC 0.971 η² 0.672 — who-has-unpaid-stock trait (like a_out_vol η²=0.74), not this-month shock. Demean Y7 0.541. |
| Q6 lead (lag1 / first-6m vs delay) | CLOSE | now 0.420 lag1 0.430 lag3 0.443 — persistence, not lead. |
| Family J merge | CLOSE | J is KEEP-Q5 diagnostic not a Y3 X. leftover after match 0.441. Do not merge J. |
| AP pending parquet column | CLOSE | no supplier-pending analogue in the store. In-memory only. Do not invent a column. |
| U-shape / all-paid dummy | CLOSE | |pending−0.30| leftover after issued_lag1 0.558 is a thin dummy; do not invent y_pending. Linear leftover still dies. |


## 1. Coverage train vs holdout; 470 NaN; ERP vs dark

Train coverage 60.3% (quote 60.3%). Holdout coverage is a check only. Never-ERP companies 470 (want 470): pending non-null 0 zero-filled 0. 470 stay NaN not 0: CONFIRM. Train NaN CM 8,395 (dark + ERP months before first issued). acf1=0.779 (quote 0.78) size ρ=0.063 (quote 0.044).

| split | cm | companies | pending nn | cov | NaN | ever-ERP / never | dark nn / zero | ERP nn / NaN | cos ever-defined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 21,157 | 1,214 | 12,762 | 60.3% | 8,395 | 744 / 470 | 0 / 0 | 12,762 / 792 | 744 |
| holdout | 1,073 | 72 | 557 | 51.9% | 516 | 40 / 32 | 0 / 0 | 557 / 25 | 40 |


| item | value |
| --- | ---: |
| train CM / companies | 21,157 / 1,214 |
| pending defined | 12,762 (60.3%) |
| train NaN CM | 8,395 |
| never-ERP companies | 470 (want 470) |
| dark pending non-null / zero-filled | 0 / 0 |
| dark 0-fill | NO — CONFIRM |
| holdout coverage (check only) | 51.9% |
| acf1 / acf3 / acf6 | 0.779 / 0.484 / 0.309 |
| size ρ vs log1p(a_in3) | 0.063 |
| size ρ vs log1p(|a_op_in|) | 0.044 (feature-report quote 0.044) |

## 2. Store vs raw pending SQL

Store vs raw pending SQL: n_both=12,762 max|Δ|=0.000e+00 mean|Δ|=0.000e+00 exact=12,762; store-only 0 sql-only 0. FORMULA MATCH — do not edit invoices.py.

SQL (same as `invoices.py` ~198–218): issued non-cancel `document_type='invoice'` with `issuance_date ≤ period_end`; unpaid `|amount|` if `payment_date` is null or after period_end; `payment_date_invalid` contributes 0 to num and den.

## 3. Spearman twins (|ρ|≥0.80)

Twins none. SIZE=False (ρ=0.063). DSO 0.492 overdue 0.017 delay 0.016 issued_lag1 -0.056 j_pay_match -0.093.

| vs | ρ | twin |ρ|≥0.80 | SIZE |ρ|≥0.50 |
| --- | --- | --- | --- |
| e_dso_proxy | 0.492 | no | — |
| e_ar_overdue | 0.017 | no | — |
| e_ar_overdue_30 | 0.124 | no | — |
| e_delay_coll | 0.016 | no | — |
| e_ar_issued | -0.023 | no | — |
| e_ar_issued_lag1 | -0.056 | no | — |
| j_pay_match | -0.093 | no | — |
| log1p(a_in3) | 0.063 | no | no |
| c_n_days_with_tx | 0.027 | no | — |
| e_credit_note_ratio | -0.102 | no | — |
| e_ar_open | 0.271 | no | — |
| e_ap_pending_share | 0.867 | (analogue) | — |
| e_ar_pending_share | 0.852 | (analogue) | — |


## 4. Single-feature train group-fold AUROC

Sign from the train side of each fold. Seed 20260918. Night Y7 issued_lag1 **0.630** (replica 0.630); TURNOVER **0.720** / B_shallow **0.712** / TURNPEND **0.7184** unchanged. Night Y3 size **0.617** (replica 0.617); days **0.711** (replica 0.711). Do not quote holdout.

Y7 pending 0.420 vs size 0.469 issued_lag1 0.630 (night 0.63) DSO 0.431. Y3 pending 0.484 vs size 0.617 days 0.711 (night 0.711). Beat-size Y7=False Y3=False.

| y | feature | n | n_pos | CV | sd | sign | folds | present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_pending_amt_share | 7,464 | 2,149 | 0.420 | 0.040 | -1 | 0.410 0.410 0.467 0.450 0.363 | 100.0% |
| y7_top1_lost | log1p(a_in3) | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 | 93.8% |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 | 100.0% |
| y7_top1_lost | e_dso_proxy | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 | 89.1% |
| y7_top1_lost | e_ar_issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 | 97.2% |
| y7_top1_lost | e_ar_overdue | 6,859 | 1,912 | 0.605 | 0.048 | 1 | 0.582 0.668 0.540 0.630 0.603 | 91.9% |
| y7_top1_lost | e_delay_coll | 5,158 | 1,304 | 0.574 | 0.041 | 1 | 0.569 0.572 0.518 0.633 0.576 | 69.1% |
| y7_top1_lost | e_credit_note_ratio | 7,308 | 2,022 | 0.545 | 0.041 | 1 | 0.505 0.546 0.552 0.513 0.608 | 97.9% |
| y7_top1_lost | j_pay_match | 6,453 | 1,775 | 0.534 | 0.028 | 1 | 0.566 0.513 0.531 0.556 0.501 | 86.5% |
| y7_top1_lost | e_ar_pending_share | 7,433 | 2,149 | 0.462 | 0.131 | 1 | 0.323 0.645 0.432 0.541 0.370 | 99.6% |
| y7_top1_lost | e_ap_pending_share | 7,376 | 2,113 | 0.450 | 0.044 | -1 | 0.433 0.462 0.507 0.459 0.388 | 98.8% |
| y3_recover_cash_6m | e_pending_amt_share | 3,446 | 239 | 0.484 | 0.102 | 1 | 0.571 0.458 0.381 0.609 0.399 | 61.0% |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 | 97.9% |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 | 100.0% |
| y3_recover_cash_6m | e_dso_proxy | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 | 42.8% |
| y3_recover_cash_6m | e_ar_issued_lag1 | 3,618 | 264 | 0.671 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 | 64.1% |
| y3_recover_cash_6m | e_ar_overdue | 2,655 | 143 | 0.616 | 0.108 | 1 | 0.776 0.591 0.558 0.661 0.493 | 47.0% |
| y3_recover_cash_6m | e_delay_coll | 1,819 | 86 | 0.512 | 0.087 | 1 | 0.426 0.455 0.626 0.582 0.473 | 32.2% |
| y3_recover_cash_6m | e_credit_note_ratio | 3,079 | 177 | 0.579 | 0.065 | -1 | 0.655 0.509 0.518 0.627 0.584 | 54.5% |
| y3_recover_cash_6m | j_pay_match | 2,646 | 151 | 0.567 | 0.130 | -1 | 0.373 0.636 0.506 0.705 0.618 | 46.8% |
| y3_recover_cash_6m | e_ar_pending_share | 2,980 | 175 | 0.471 | 0.118 | 1 | 0.331 0.477 0.606 0.566 0.377 | 52.8% |
| y3_recover_cash_6m | e_ap_pending_share | 3,428 | 239 | 0.406 | 0.026 | 1 | 0.418 0.440 0.397 0.404 0.369 | 60.7% |


## 5. Honest leftover after the bar

Y7 bar = DSO and/or issued_lag1. Y3 bar = days. Leftover <0.55 dies. OLS residual of pending on the bar; rank-residual is the orthogonal-to-bar twin.

Y7 leftover after DSO 0.421 (dies <0.55); after issued_lag1 0.418 (dies <0.55); after both 0.419. Y3 leftover after days 0.443 (dies <0.55). Rank-ortho DSO 0.461 issued 0.424 days 0.543.

| y | residual | n | n_pos | CV | R² | folds |
| --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | after DSO | 6,651 | 1,623 | 0.421 | 0.001 | 0.415 0.421 0.475 0.448 0.345 |
| y7_top1_lost | after issued_lag1 | 7,253 | 2,072 | 0.418 | 0.000 | 0.417 0.411 0.475 0.440 0.347 |
| y7_top1_lost | after issued | 7,464 | 2,149 | 0.418 | 0.000 | 0.409 0.409 0.468 0.440 0.363 |
| y7_top1_lost | after overdue | 6,859 | 1,912 | 0.490 | 0.002 | 0.421 0.566 0.525 0.529 0.408 |
| y7_top1_lost | after DSO+issued_lag1 | 6,470 | 1,565 | 0.419 | 0.001 | 0.423 0.421 0.482 0.444 0.324 |
| y7_top1_lost | after size | 7,000 | 1,987 | 0.422 | 0.000 | 0.432 0.417 0.477 0.451 0.333 |
| y7_top1_lost | after days | 7,464 | 2,149 | 0.421 | 0.000 | 0.411 0.411 0.467 0.453 0.363 |
| y7_top1_lost | rank-resid after DSO | 6,651 | 1,623 | 0.461 | — | 0.451 0.473 0.513 0.477 0.391 |
| y7_top1_lost | rank-resid after issued_lag1 | 7,253 | 2,072 | 0.424 | — | 0.422 0.418 0.481 0.459 0.341 |
| y3_recover_cash_6m | after days | 3,446 | 239 | 0.443 | 0.000 | 0.570 0.451 0.382 0.419 0.395 |
| y3_recover_cash_6m | after size | 3,383 | 233 | 0.536 | 0.000 | 0.580 0.460 0.623 0.621 0.395 |
| y3_recover_cash_6m | after days+size | 3,383 | 233 | 0.544 | 0.000 | 0.581 0.471 0.623 0.643 0.401 |
| y3_recover_cash_6m | after DSO | 2,418 | 93 | 0.538 | 0.001 | 0.572 0.547 0.610 0.530 0.431 |
| y3_recover_cash_6m | after issued_lag1 | 3,446 | 239 | 0.535 | 0.000 | 0.571 0.458 0.619 0.630 0.400 |
| y3_recover_cash_6m | rank-resid after days | 3,446 | 239 | 0.543 | — | 0.580 0.468 0.622 0.640 0.407 |


## 6. Twin screen — DROP the weaker twin

No |ρ|≥0.80 twin.

_(no |ρ|≥0.80 twin)_


## 7. SIZE terciles and invoice-book-only

Y7 all-NaN-dropped 0.420 vs book-only 0.420 (470 never enter the defined set — same rows). Size terciles T1/T2/T3 0.425 / 0.403 / 0.482.

| slice | n | n_pos | CV | folds |
| --- | --- | --- | --- | --- |
| all-train NaN-dropped (labeled) | 7,464 | 2,149 | 0.420 | 0.410 0.410 0.467 0.450 0.363 |
| invoice-book-only (drop 470) | 7,464 | 2,149 | 0.420 | 0.410 0.410 0.467 0.450 0.363 |
| size T1 small | 2,334 | 755 | 0.425 | 0.406 0.475 0.495 0.417 0.334 |
| size T2 mid | 2,333 | 648 | 0.403 | 0.436 0.332 0.456 0.462 0.327 |
| size T3 large | 2,333 | 584 | 0.482 | 0.526 0.442 0.528 0.523 0.393 |
| Y3 invoice-book-only | 3,446 | 239 | 0.484 | 0.571 0.458 0.381 0.609 0.399 |


## 8. Q6 — lag1 / lag3, empty-on-short, delay first 6 months

Delay is empty on 2024-09..2025-02 (`DELAY_MASK_BEFORE`). Pending is a stock of unpaid issued — it can be defined as soon as the company has issued.

Q6 Y7 pending now 0.420 lag1 0.430 lag3 0.443. lag1 ≈ now — persistent trait, not a new lead. First 6 calendar months: pending cov 58.3% vs delay 0.0% (pending populated earlier — unpaid stock, not paid-delay window). Short so-far 0.447 long 0.449.

| window | cm | pending nn | pending cov | delay nn | delay cov |
| --- | --- | --- | --- | --- | --- |
| calendar first 6m (2024-09..2025-02) | 3,229 | 1,884 | 58.3% | 0 | 0.0% |
| so-far <6 months | 6,068 | 3,196 | 52.7% | 707 | 11.7% |
| calendar after delay-ok | 17,928 | 10,878 | 60.7% | 6,752 | 37.7% |
| so-far ≥6 | 15,089 | 9,566 | 63.4% | 6,045 | 40.1% |


| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_pending_amt_share | 7,464 | 2,149 | 0.420 | 0.040 | -1 | 0.410 0.410 0.467 0.450 0.363 |
| y3_recover_cash_6m | e_pending_amt_share | 3,446 | 239 | 0.484 | 0.102 | 1 | 0.571 0.458 0.381 0.609 0.399 |
| y7_top1_lost | e_pending_amt_share_lag1 | 7,201 | 2,052 | 0.430 | 0.042 | -1 | 0.422 0.435 0.476 0.452 0.365 |
| y3_recover_cash_6m | e_pending_amt_share_lag1 | 3,397 | 237 | 0.534 | 0.088 | 1 | 0.544 0.467 0.627 0.608 0.423 |
| y7_top1_lost | e_pending_amt_share_lag3 | 6,331 | 1,775 | 0.443 | 0.054 | -1 | 0.447 0.472 0.477 0.468 0.348 |
| y3_recover_cash_6m | e_pending_amt_share_lag3 | 3,011 | 209 | 0.410 | 0.069 | 1 | 0.527 0.384 0.411 0.349 0.381 |
| y7_top1_lost | e_delay_coll | 5,158 | 1,304 | 0.574 | 0.041 | 1 | 0.569 0.572 0.518 0.633 0.576 |
| y3_recover_cash_6m | e_delay_coll | 1,819 | 86 | 0.512 | 0.087 | 1 | 0.426 0.455 0.626 0.582 0.473 |
| y7_top1_lost | e_ar_issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y3_recover_cash_6m | e_ar_issued_lag1 | 3,618 | 264 | 0.671 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 |
| y7_top1_lost | pending short_<12 so-far | 4,421 | 1,337 | 0.447 | 0.069 | 1 | 0.374 0.409 0.507 0.534 0.414 |
| y7_top1_lost | pending long_>=18 so-far | 916 | 209 | 0.449 | 0.082 | -1 | 0.476 0.512 0.464 0.490 0.306 |
| y7_top1_lost | pending calendar first 6m | 1,071 | 393 | 0.437 | 0.125 | 1 | 0.229 0.546 0.502 0.487 0.420 |


## 9. vs Family J `j_pay_match` (in-memory, not merged)

J is KEEP-Q5 diagnostic, not a Y3 X. Do not merge J.

ρ pending vs j_pay_match -0.093 twin=False. J is KEEP-Q5 diagnostic not a Y3 X — do not merge J. Y7 leftover after match-rate 0.441 (dies <0.55). J Y7 0.534 Y3 0.567. j_has_book=0 non-null=0 (want 0).

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | j_pay_match | 6,453 | 1,775 | 0.534 | 0.028 | 1 | 0.566 0.513 0.531 0.556 0.501 |
| y7_top1_lost | pending resid after j_pay_match | 6,453 | 1,775 | 0.441 | 0.061 | -1 | 0.490 0.429 0.502 0.434 0.349 |
| y3_recover_cash_6m | j_pay_match | 2,646 | 151 | 0.567 | 0.130 | -1 | 0.373 0.636 0.506 0.705 0.618 |
| y3_recover_cash_6m | pending resid after j_pay_match | 2,646 | 151 | 0.357 | 0.036 | 1 | 0.321 0.415 0.339 0.363 0.347 |


## 10. vs credit-note and delay (siblings read-only)

credit_note_qa.md published leftover after issued_lag1 **0.597** (KEEP Y7 leftover; DROP as Y3 X). delay_qa.md published: delay leftover after DSO **0.581** / issued_lag1 **0.583** (KEEP Y7 leftover; DROP as Y3 X). Not racing the sibling script. Pending leftover after CN 0.427; after delay 0.449 (dies <0.55 if below chance).

| residual | Y7 CV | Y3 CV | R² | Y7 folds |
| --- | --- | --- | --- | --- |
| after e_credit_note_ratio | 0.427 | 0.505 | 0.002 | 0.429 0.417 0.464 0.455 0.369 |
| after e_delay_coll | 0.449 | 0.592 | 0.003 | 0.512 0.373 0.529 0.516 0.314 |
| after e_ar_overdue_30 | 0.488 | 0.551 | 0.016 | 0.426 0.552 0.521 0.525 0.415 |


## 11. ICC / company-demean — trait vs month shock

ICC=0.971 (quote 0.97) η²=0.672 on 744 companies / 12,762 CM. Group ICC=0.987 η²=0.484. a_out_vol η²=0.74. TRAIT like a_out_vol (who-has-unpaid-stock), not a month shock. Y7 demean leftover 0.541 Y3 0.538.

| item | value |
| --- | ---: |
| company ICC / η² | 0.971 / 0.672 |
| companies / CM | 744 / 12,762 |
| group ICC / η² | 0.987 / 0.484 |
| a_out_vol η² (quote) | 0.74 |
| trait (ICC≥0.85) | True |
| Y7 / Y3 demean leftover | 0.541 / 0.538 |

## 12. Brief map

See PARK / CLOSE / KEEP above. Night quotes unchanged.

## Extra A. Pending vs overdue pile

Median split pending 0.333 × overdue 0.818. High-pending & low-overdue (stock unpaid, not yet 30d/due-late) Y7 rate 23.0% n=1,641; both-high Y7 rate 36.3%.

| pile | n | Y7 rate | median pending | median overdue |
| --- | --- | --- | --- | --- |
| low-pending high-overdue | 1,641 | 31.1% | 0.147 | 1.000 |
| high-pending low-overdue | 1,641 | 23.0% | 0.742 | 0.349 |
| both high | 1,789 | 36.3% | 0.785 | 0.999 |
| both low | 1,788 | 20.9% | 0.146 | 0.291 |


## Extra B. AP / supplier pending analogue

No supplier-pending column in monthly.parquet (has_col=False). Do not invent one. In-memory AP-only share cov 59.6% ρ vs mixed pending 0.867; AR-only ρ 0.852. Y7 AP-only 0.450 AR-only 0.462.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_ar_pending_share (in-memory) | 7,433 | 2,149 | 0.462 | 0.131 | 1 | 0.323 0.645 0.432 0.541 0.370 |
| y7_top1_lost | e_ap_pending_share (in-memory) | 7,376 | 2,113 | 0.450 | 0.044 | -1 | 0.433 0.462 0.507 0.459 0.388 |
| y3_recover_cash_6m | e_ap_pending_share (in-memory) | 3,428 | 239 | 0.406 | 0.026 | 1 | 0.418 0.440 0.397 0.404 0.369 |


## Extra C. Holdout coverage only (no AUROC)

Holdout coverage only: 557/1,073 = 51.9% on 72 companies. Dark holdout CM 491 pending-nn 0 (want 0). No AUROC, no percentiles, no fit on the 72.

## Extra D. Quintiles of pending vs Y7 rate

Y7 rates by pending quintile: 35.6%, 23.4%, 24.3%, 27.4%, 33.3%. U-shape.

| q | bin | n | Y7 rate | median pending |
| --- | --- | --- | --- | --- |
| Q1 | (-0.001, 0.0894] | 1,493 | 35.6% | 0.037 |
| Q2 | (0.0894, 0.225] | 1,493 | 23.4% | 0.154 |
| Q3 | (0.225, 0.42] | 1,492 | 24.3% | 0.304 |
| Q4 | (0.42, 0.84] | 1,493 | 27.4% | 0.595 |
| Q5 | (0.84, 1.0] | 1,493 | 33.3% | 0.999 |


Plot: `pending_qa.png`.

## Extra E. U-shape probe (not a Y)

U-shape probe: |pending−0.30| Y7 0.562 tail-dummy 0.555. Leftover of the U after issued_lag1 0.558 R²=0.000 (lives as a nonlinear dummy). Do not invent y_pending from the tails.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | |pending-0.30| | 7,464 | 2,149 | 0.562 | 0.066 | 1 | 0.668 0.497 0.541 0.522 0.580 |
| y7_top1_lost | tail dummy Q1 or Q5 | 7,464 | 2,149 | 0.555 | 0.074 | 1 | 0.623 0.489 0.542 0.481 0.641 |
| y7_top1_lost | |pending-0.30| resid after issued_lag1 | 7,253 | 2,072 | 0.558 | 0.068 | 1 | 0.667 0.495 0.536 0.515 0.575 |


## Extra F. Fold 4 (issued owns 0.647 / TURNOVER 0.680)

Fold 4 pending 0.363 vs issued_lag1 0.647 vs TURNOVER 0.680. issued owns fold 4. Hard groups ('GROUP_0222', 'GROUP_0108'): pending medians 0.062, 0.013.

| feature | fold4 | CV | folds |
| --- | --- | --- | --- |
| e_pending_amt_share | 0.363 | 0.420 | 0.410 0.410 0.467 0.450 0.363 |
| e_ar_issued_lag1 | 0.647 | 0.630 | 0.643 0.662 0.590 0.605 0.647 |
| e_dso_proxy | 0.342 | 0.431 | 0.370 0.600 0.420 0.424 0.342 |
| e_delay_coll | 0.576 | 0.574 | 0.569 0.572 0.518 0.633 0.576 |
| e_ar_overdue | 0.603 | 0.605 | 0.582 0.668 0.540 0.630 0.603 |


| group | n | pos | Y7 rate | median pending |
| --- | --- | --- | --- | --- |
| GROUP_0222 | 336 | 241 | 71.7% | 0.062 |
| GROUP_0108 | 204 | 130 | 63.7% | 0.013 |


## Extra G. New-book vs all-paid tails

New-book: first-3-months Y7 rate 33.7% vs mature ≥6m 27.3%. pending=0 (all paid) Y7 31.9%; pending=1 (nothing paid) 26.7%. Q1 tail is the emptied book (all collected), not unpaid stock. ERP months before first issued (pending NaN on ever-ERP): 792.

| slice | n | pos | Y7 rate | median pending |
| --- | --- | --- | --- | --- |
| first issued month | 172 | 65 | 37.8% | 0.325 |
| months_on_book 1–3 | 709 | 239 | 33.7% | 0.330 |
| months_on_book ≥6 | 2,937 | 803 | 27.3% | 0.286 |
| pending == 0 (all paid) | 138 | 44 | 31.9% | 0.000 |
| pending == 1 (nothing paid) | 574 | 153 | 26.7% | 1.000 |
| 0 < pending < 1 | 3,596 | 1,051 | 29.2% | 0.258 |


## Extra H. Company-mean trait vs Y7 rate

Company-mean pending vs company Y7 rate ρ=0.022 on 631 companies (≥3 labeled months). Ever-lost AUROC 0.520. Size ρ vs log1p(|a_op_in|) 0.044 (feature-report quote 0.044). Company mean is not a Y7 ranker.

## Extra I. DSO clip 24m leftover (Y7 card clip)

DSO clipped at 24m: ρ vs pending 0.492 (raw Spearman was ~0.49). Y7 leftover after clipped DSO 0.448 R²=0.130; clipped DSO single 0.391. still dies.

| residual | n | n_pos | CV | R² | folds |
| --- | --- | --- | --- | --- | --- |
| after DSO clip 24m | 6,651 | 1,623 | 0.448 | 0.130 | 0.443 0.472 0.509 0.467 0.350 |


## Extra J. All-paid dummy (Q1 emptied book)

All-paid dummy Y7 0.460; leftover after issued_lag1 0.572 R²=0.000; after DSO 0.548 R²=0.000. thin dummy leftover. Not a reason to KEEP pending as X.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | all-paid dummy (pending≤0.09) | 7,464 | 2,149 | 0.460 | 0.063 | 1 | 0.521 0.444 0.507 0.466 0.361 |
| y7_top1_lost | all-paid resid after issued_lag1 | 7,253 | 2,072 | 0.572 | 0.085 | -1 | 0.573 0.670 0.574 0.603 0.437 |
| y7_top1_lost | all-paid resid after DSO | 6,651 | 1,623 | 0.548 | 0.063 | 1 | 0.589 0.489 0.572 0.473 0.617 |


## Extra K. AR / AP mix inside the blended share

Mixed pending is a blend: n both AR+AP 10,659; AR-only 158; AP-only 1,945. Mean |mixed−AR|=0.103 |mixed−AP|=0.115 — closer to AR. ρ AR 0.852 / AP 0.867. No AP column in parquet; do not invent one.

## Extra L. Same-n leftover vs issued_lag1

Same-n n=7,253: pending 0.420 vs issued_lag1 0.630 (night 0.630). Residual 0.418 lift vs issued -0.211. Residualizing does not create a new object.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | pending same-n | 7,253 | 2,072 | 0.420 | 0.048 | -1 | 0.417 0.411 0.474 0.448 0.347 |
| y7_top1_lost | issued_lag1 same-n | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y7_top1_lost | pending resid same-n | 7,253 | 2,072 | 0.418 | 0.047 | -1 | 0.417 0.411 0.475 0.440 0.347 |


## Extra M. Inverse leftover — delay after pending

Inverse leftover: delay after pending 0.582 R²=0.003 (sibling KEEP 0.581 should survive). Y7 labeled on dark 470: 0 (want 0 — Y7 is invoice-built).

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | delay resid after pending | 5,158 | 1,304 | 0.582 | 0.054 | 1 | 0.565 0.546 0.522 0.636 0.640 |


## What failed / next (held for wave note)

- Y7 leftover after DSO 0.421 / issued_lag1 0.418 dies <0.55 — unused stock; TURNPEND 0.7184 already lost
- Y3 leftover after days 0.443 dies — stays off the 15-col card
- U-shape / all-paid dummy leftover is thin (0.558 / 0.572) — CLOSE, do not invent y_pending
- Fold 4 pending 0.363 vs issued 0.647 — issued owns the hard fold

Elapsed 10s. Cuts 1–12 plus extras (pile, AP analogue, holdout coverage, quintiles, U-shape, fold 4, new-book, company-mean, DSO clip, all-paid dummy, AR/AP mix, same-n).

## What this module did not do

- Did not change night Y3 0.762 / 0.752 or Y7 TURNOVER 0.720 / B_shallow 0.712 / TURNPEND 0.7184.
- Did not put pending on the Y3 15-col card. Did not grow TURNOVER. Did not use Family D as Y7 X.
- Did not use Family E as Y5 X. Did not invent `y_pending`. Did not write 0–100 / pillars.
- Did not touch `product/`. Did not edit `invoices.py` / `gbm_y7_core.py` / sibling QA scripts.
- Did not rewrite parquet or duckdb. Did not merge Family I/M/J. Did not run `build_targets`.
- Did not fit on holdout 72. Did not commit. Did not write the parent journal / LIVE / canvas.
