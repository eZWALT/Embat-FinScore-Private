# Q4/Q5 credit-note leftover after issued_lag1

Generated `2026-09-19T04:33:47+02:00` by agent `0c3bf32d`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_credit_note`. Night Y7 quote stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. Y7 never D. Y3 never B. CN stays off the 15-col card.

`e_credit_note_ratio` = |credit notes| / (|invoices| + |credit notes|) issued this period. Type `credit_note` if present; else ERP stand-ins `note` and `refund`.

## Headline

CN defined 53.0% of train CM; dark 470 is NaN not 0 (COMP_0962 refund-only is the one defined dark month). No canonical `credit_note` — fallback **note 98.6% / refund 1.4%**. ρ vs issued_lag1 0.237 vs DSO -0.050 (not a twin). Y7 CN 0.545 vs issued_lag1 0.630 (night 0.630); leftover after issued_lag1 0.597 (same-n lift 0.049; demean leftover 0.509 dies — who-uses-notes). Y3 CN 0.579 vs days 0.711 leftover 0.560. Fold 4 CN 0.608 vs issued 0.647 — TURNOVER 0.680 is issued. Y7 leftover **KEEP**. Y3 X **CLOSE / DROP from the 44**. PARK as health Y. Do not grow TURNOVER 0.720.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_credit_note`. Dark 470 = NaN, not 0 (COMP_0962 refund-only exception). |
| 2 | Who is improving? | Not this ratio. |
| 3 | Who is turning? | CN lag1 is a weak Q6, not a turn clock. |
| 4 | Dip vs fall? | Y7 leftover after issued_lag1 **KEEP** — residual after issued_lag1 0.597 beats chance and is not a twin (ρ issued_lag1 0.237, DSO -0.050). Still do not grow TURNOVER 0.720. |
| 5 | Why did it change? | **KEEP-Q5 footnote** — who-uses-notes leftover after issued — not this month’s correction, not thin issuance. Fold 4: issued owns fold 4. |
| 6 | Months earlier? | CN lag1 short 0.542 vs issued_lag1 0.630 — **CLOSE**. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| CN as Y7 leftover after issued_lag1 | **KEEP** | residual after issued_lag1 0.597 beats chance and is not a twin (ρ issued_lag1 0.237, DSO -0.050). Still do not grow TURNOVER 0.720. |
| CN as Y7 add-on / grow TURNOVER | **CLOSE** | do not grow TURNOVER; night quote stays 0.720 / 0.712 |
| CN as Y3 X / the 44 | **CLOSE / DROP from the 44** | Y3 CN 0.579 loses to days 0.711 (leftover after days 0.560). |
| CN as a health Y | **PARK** | do not invent `y_credit_note` |
| Q5 footnote without “thin issuance” | **KEEP-Q5 footnote** | who-uses-notes leftover after issued — not this month’s correction, not thin issuance |
| Q6 CN lag1 | **CLOSE** | Y7 CN lag1 short so-far 0.542 (q6_quoted 0.542, CONFIRM). CLOSE as Q6 — short lag1 stays weak vs issued_lag1 0.630. |
| Fold 4 Y7 | **issued owns fold 4** | CN 0.608 vs issued_lag1 0.647 vs TURNOVER 0.680 |

## 1. Prevalence / coverage (dark = NaN not 0)

Train CN defined 11,207/21,157 (53.0%); companies ever-defined 744/1214. Among defined: zero 57.7% pos 42.3%; modal 0 share 57.7%. Dark never-ERP companies 470 (want 470): CN non-null 1 zero-filled 0 (CONFIRM no 0-fill). Ever-ERP 744 CM-defined 82.7%. Y7 labeled 7,464 pos 2,149 CN-nn 7,308; Y3 stressed 5,648 pos 402 CN-nn 3,079. acf1=-0.088 acf3=-0.091.

| item | value |
| --- | ---: |
| train CM / companies | 21,157 / 1,214 |
| CN defined | 11,207 (53.0%) |
| among defined: zero / >0 | 57.7% / 42.3% |
| modal / modal share | 0 / 57.7% |
| ever-ERP / never-ERP | 744 / 470 |
| dark CN non-null / zero-filled | 1 / 0 |
| dark 0-fill | NO — CONFIRM (nn=1) |
| Y7 labeled / pos / CN-nn | 7,464 / 2,149 / 7,308 |
| Y3 labeled / pos / CN-nn | 5,648 / 402 / 3,079 |
| acf1 / acf3 | -0.088 / -0.091 |
| CN defined on issued=0 ERP months | 2,623 / 4,971 |

## 2. Formula vs raw invoice types

Raw document_type values: ['cheque', 'deliveryNote', 'deposit', 'invoice', 'invoiceGroup', 'note', 'other', 'paymentDocument', 'purchaseOrder', 'refund']. Feature uses **fallback** ('note', 'refund'). Store vs recompute: n_both=11,207 max|Δ|=7.772e-16 mean|Δ|=8.433e-18 exact 11,207; store-only 0 recon-only 0. FORMULA MATCH.

| document_type | n_docs | abs_amt | n_cm | in_feature |
| --- | --- | --- | --- | --- |
| invoice | 709,267 | 191,856,021,940 | 12,893 | invoice |
| note | 32,909 | 78,738,882,230 | 4,848 | yes |
| refund | 1,931 | 1,126,601,394 | 322 | yes |
| paymentdocument | 51,564 | 1,049,957,997 | 2,176 | no |
| invoicegroup | 13,607 | 625,130,251 | 1,216 | no |
| deposit | 7,977 | 252,646,597 | 455 | no |
| deliverynote | 3,639 | 5,057,016 | 38 | no |
| purchaseorder | 791 | 4,225,254 | 77 | no |
| other | 718 | 3,199,312 | 40 | no |
| cheque | 11 | 1,717 | 4 | no |


## 3. Spearman twins (|ρ|≥0.80)

CN vs issued 0.251, issued_lag1 0.237, DSO -0.050, size 0.209, days 0.317. Not a |ρ|≥0.80 twin of issued / issued_lag1 / DSO. Not a size clone.

| vs | ρ | twin |ρ|≥0.80 |
| --- | --- | --- |
| e_ar_issued | 0.251 | no |
| e_ar_issued_lag1 | 0.237 | no |
| e_ar_issued_lag_cv | 0.090 | no |
| e_dso_proxy | -0.050 | no |
| log1p(a_in3) | 0.209 | no |
| c_n_days_with_tx | 0.317 | no |
| e_credit_note_ratio_lag1 | 0.619 | no |
| e_ar_issued_lag3 | 0.218 | no |


## 4. Single-feature train group-fold AUROC

Sign from the train side of each fold. Seed 20260918. Night Y7 issued_lag1 **0.630** (replica 0.630); TURNOVER **0.720** / B_shallow **0.712** unchanged. Night Y3 size **0.617** (replica 0.617); days **0.711** (replica 0.711).

Y7 CN 0.545 vs issued_lag1 0.630 (night 0.630, Δ -0.000) vs DSO 0.431 vs issued-lag CV 0.627. CN lag1 0.548. Quote TURNOVER 0.720 / B_shallow 0.712 unchanged. Y3 CN 0.579 vs size 0.617 (night 0.617) vs days 0.711 (night 0.711, Δ 0.000). CN loses to issued_lag1. CN loses to days — expect CLOSE as Y3 X.

| y | feature | n | n_pos | CV | sd | sign | folds | present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | e_credit_note_ratio | 7,308 | 2,022 | 0.545 | 0.041 | 1 | 0.505 0.546 0.552 0.513 0.608 | 97.9% |
| y7_top1_lost | e_credit_note_ratio_lag1 | 7,071 | 1,969 | 0.548 | 0.040 | 1 | 0.505 0.550 0.550 0.524 0.612 | 94.7% |
| y7_top1_lost | e_credit_note_ratio_lag3 | 6,188 | 1,710 | 0.552 | 0.041 | 1 | 0.515 0.546 0.554 0.525 0.620 | 82.9% |
| y7_top1_lost | e_ar_issued | 7,464 | 2,149 | 0.663 | 0.037 | -1 | 0.660 0.702 0.615 0.641 0.696 | 100.0% |
| y7_top1_lost | e_ar_issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 | 97.2% |
| y7_top1_lost | e_ar_issued_lag_cv | 7,253 | 2,072 | 0.627 | 0.030 | 1 | 0.609 0.666 0.600 0.654 0.605 | 97.2% |
| y7_top1_lost | e_dso_proxy | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 | 89.1% |
| y7_top1_lost | log1p_a_in3 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 | 93.8% |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 | 100.0% |
| y3_recover_cash_6m | e_credit_note_ratio | 3,079 | 177 | 0.579 | 0.065 | -1 | 0.655 0.509 0.518 0.627 0.584 | 54.5% |
| y3_recover_cash_6m | e_credit_note_ratio_lag1 | 3,063 | 181 | 0.547 | 0.056 | -1 | 0.624 0.541 0.472 0.573 0.525 | 54.2% |
| y3_recover_cash_6m | e_credit_note_ratio_lag3 | 2,732 | 159 | 0.568 | 0.073 | -1 | 0.644 0.541 0.459 0.618 0.577 | 48.4% |
| y3_recover_cash_6m | e_ar_issued | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 | 64.1% |
| y3_recover_cash_6m | e_ar_issued_lag1 | 3,618 | 264 | 0.671 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 | 64.1% |
| y3_recover_cash_6m | e_ar_issued_lag_cv | 3,618 | 264 | 0.571 | 0.062 | -1 | 0.614 0.654 0.513 0.560 0.516 | 64.1% |
| y3_recover_cash_6m | e_dso_proxy | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 | 42.8% |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 | 97.9% |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 | 100.0% |


## 5. Residual Y7 after issued_lag1 / DSO / issued

KEEP leftover only if residual after issued_lag1 beats chance (≥0.55) **and** is not a twin. If leftover dies, CN on TURNOVER is a twin — CLOSE as add-on, do not grow the card.

Y7 leftover after issued_lag1 0.597 (lives ≥0.55); after issued 0.600; after DSO 0.479; after both 0.577. Leftover is sayable without thin issuance.

| residual | n | n_pos | CV | R² | slope | folds |
| --- | --- | --- | --- | --- | --- | --- |
| after issued_lag1 | 7,105 | 1,951 | 0.597 | 0.002 | 0.000 | 0.607 0.575 0.631 0.542 0.631 |
| after issued | 7,308 | 2,022 | 0.600 | 0.002 | 0.000 | 0.599 0.578 0.630 0.552 0.641 |
| after DSO | 6,651 | 1,623 | 0.479 | 0.002 | 0.000 | 0.465 0.556 0.506 0.509 0.357 |
| after issued_lag_cv | 7,105 | 1,951 | 0.464 | 0.000 | 0.003 | 0.434 0.496 0.500 0.478 0.414 |
| after days | 7,308 | 2,022 | 0.547 | 0.004 | 0.001 | 0.467 0.526 0.545 0.548 0.648 |
| after size | 6,860 | 1,871 | 0.561 | 0.006 | 0.002 | 0.476 0.560 0.571 0.534 0.661 |
| after issued_lag1+DSO | 6,470 | 1,565 | 0.577 | 0.007 | 0.000 | 0.532 0.580 0.579 0.536 0.656 |


## 6. Residual Y3 after days

Y3 leftover after days 0.560 vs size 0.462. Expect CLOSE as Y3 X — leftover after days dies.

| residual | n | n_pos | CV | R² | folds |
| --- | --- | --- | --- | --- | --- |
| after days | 3,079 | 177 | 0.560 | 0.004 | 0.573 0.580 0.562 0.535 0.552 |
| after size | 3,020 | 171 | 0.462 | 0.006 | 0.514 0.465 0.422 0.453 0.459 |
| after issued_lag1 | 3,079 | 177 | 0.489 | 0.002 | 0.543 0.469 0.478 0.409 0.547 |
| after DSO | 2,418 | 93 | 0.639 | 0.002 | 0.797 0.621 0.584 0.668 0.523 |
| after days+size | 3,020 | 171 | 0.553 | 0.007 | 0.557 0.580 0.603 0.512 0.514 |


## 7. Token mix — credit_note vs note vs refund

CN-like |amt| mass (train, 2024-09..2026-08): canonical 0.0%, note 98.6%, refund 1.4%. Feature kind=fallback uses ('note', 'refund'); used-mass 79,865,483,625 of CN-like 79,865,483,625. Mostly `note` stand-ins.

| token | n_docs | n_co | abs_amt | share_mass | in_feature |
| --- | --- | --- | --- | --- | --- |
| credit_note | 0 | 0 | 0 | 0.0% | no |
| creditnote | 0 | 0 | 0 | 0.0% | no |
| note | 32,909 | 470 | 78,738,882,230 | 98.6% | yes |
| refund | 1,931 | 63 | 1,126,601,394 | 1.4% | yes |


## 8. Q6 — lag1 / lag3 on short vs long

Y7 CN lag1 short so-far 0.542 (q6_quoted 0.542, CONFIRM). CLOSE as Q6 — short lag1 stays weak vs issued_lag1 0.630.

| slice | col | n_nn | n_pos | present | CV | Δ night 0.630 |
| --- | --- | --- | --- | --- | --- | --- |
| all | e_credit_note_ratio | 7,308 | 2,022 | 97.9% | 0.545 | -0.085 |
| all | e_credit_note_ratio_lag1 | 7,071 | 1,969 | 94.7% | 0.548 | -0.082 |
| all | e_credit_note_ratio_lag3 | 6,188 | 1,710 | 82.9% | 0.552 | -0.078 |
| all | e_ar_issued_lag1 | 7,253 | 2,072 | 97.2% | 0.630 | -0.000 |
| short_<12_sofar | e_credit_note_ratio | 4,323 | 1,262 | 97.8% | 0.544 | -0.086 |
| short_<12_sofar | e_credit_note_ratio_lag1 | 4,086 | 1,198 | 92.4% | 0.542 | -0.088 |
| short_<12_sofar | e_credit_note_ratio_lag3 | 3,242 | 936 | 73.3% | 0.548 | -0.082 |
| short_<12_sofar | e_ar_issued_lag1 | 4,210 | 1,260 | 95.2% | 0.626 | -0.004 |
| long_>=18_sofar | e_credit_note_ratio | 906 | 199 | 98.9% | 0.539 | -0.091 |
| long_>=18_sofar | e_credit_note_ratio_lag1 | 908 | 203 | 99.1% | 0.555 | -0.075 |
| long_>=18_sofar | e_credit_note_ratio_lag3 | 898 | 201 | 98.0% | 0.558 | -0.072 |
| long_>=18_sofar | e_ar_issued_lag1 | 916 | 209 | 100.0% | 0.646 | 0.016 |
| short_<12_company | e_credit_note_ratio | 765 | 167 | 98.8% | 0.474 | -0.156 |
| short_<12_company | e_credit_note_ratio_lag1 | 641 | 131 | 82.8% | 0.549 | -0.081 |
| short_<12_company | e_credit_note_ratio_lag3 | 357 | 60 | 46.1% | 0.472 | -0.158 |
| short_<12_company | e_ar_issued_lag1 | 666 | 140 | 86.0% | 0.722 | 0.092 |
| long_>=18_company | e_credit_note_ratio | 6,091 | 1,735 | 97.8% | 0.556 | -0.074 |
| long_>=18_company | e_credit_note_ratio_lag1 | 6,013 | 1,725 | 96.5% | 0.560 | -0.070 |
| long_>=18_company | e_credit_note_ratio_lag3 | 5,505 | 1,560 | 88.4% | 0.558 | -0.072 |
| long_>=18_company | e_ar_issued_lag1 | 6,154 | 1,813 | 98.8% | 0.616 | -0.014 |


## 9. Fold 4 Y7 (DSO failed; TURNOVER 0.680)

Fold 4 singles (sign from folds 0–3): CN 0.608, issued_lag1 0.647, CN residual 0.631. TURNOVER fold-4 quote 0.680. CN does not own fold 4.

OOF fold bits (sign per fold):

| feature | CV | fold4 | folds |
| --- | --- | --- | --- |
| e_credit_note_ratio | 0.545 | 0.608 | 0.505 0.546 0.552 0.513 0.608 |
| e_credit_note_ratio_lag1 | 0.548 | 0.612 | 0.505 0.550 0.550 0.524 0.612 |
| e_ar_issued_lag1 | 0.630 | 0.647 | 0.643 0.662 0.590 0.605 0.647 |
| e_ar_issued_lag_cv | 0.627 | 0.605 | 0.609 0.666 0.600 0.654 0.605 |
| e_dso_proxy | 0.431 | 0.342 | 0.370 0.600 0.420 0.424 0.342 |
| CN resid after issued_lag1 | 0.597 | 0.631 | 0.607 0.575 0.631 0.542 0.631 |


Fold 4 only (sign from folds 0–3):

| feature | n_va | n_pos | fold4 | sign |
| --- | --- | --- | --- | --- |
| e_credit_note_ratio | 1768 | 668 | 0.608 | 1 |
| e_credit_note_ratio_lag1 | 1700 | 639 | 0.612 | 1 |
| e_ar_issued_lag1 | 1734 | 657 | 0.647 | -1 |
| e_ar_issued_lag_cv | 1734 | 657 | 0.605 | 1 |
| e_dso_proxy | 1553 | 528 | 0.342 | 1 |
| CN resid after issued_lag1 | 1707 | 639 | 0.631 | 1 |


Hard groups (y7_core):

| group | n_lab | n_pos | rate | CN p50 | issued_lag1 p50 |
| --- | --- | --- | --- | --- | --- |
| GROUP_0222 | 336 | 241 | 71.7% | 0.144 | 40,714 |
| GROUP_0108 | 204 | 130 | 63.7% | 0.021 | 5,203 |


## 10. ICC / company-demean

CN ICC=0.828 (CONFIRM feature-report 0.83) k=744. Y7 demean 0.508 company-mean 0.566. Company-mean carries the skill — style dummy.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | CN raw | 7,308 | 2,022 | 0.545 | 0.041 | 1 | 0.505 0.546 0.552 0.513 0.608 |
| y7_top1_lost | CN demean | 7,308 | 2,022 | 0.508 | 0.052 | -1 | 0.531 0.423 0.525 0.561 0.501 |
| y7_top1_lost | CN company-mean | 7,464 | 2,149 | 0.566 | 0.074 | 1 | 0.541 0.463 0.565 0.596 0.664 |
| y3_recover_cash_6m | CN raw | 3,079 | 177 | 0.579 | 0.065 | -1 | 0.655 0.509 0.518 0.627 0.584 |
| y3_recover_cash_6m | CN demean | 3,079 | 177 | 0.548 | 0.027 | -1 | 0.570 0.506 0.571 0.546 0.546 |
| y3_recover_cash_6m | CN company-mean | 3,618 | 264 | 0.490 | 0.035 | -1 | 0.519 0.527 0.467 0.442 0.495 |


## 11. Dark 470 + holdout coverage (no AUROC)

Holdout 72 coverage only: Y7 labeled pos=122 (CONFIRM 122) — not a trophy AUROC. Dark holdout companies 32 (join-QA 32): CN nn=0 (NaN stays). Do not fill 0.

| slice | n_cm | n_co | CN nn | share_nn | Y7 lab / pos | Y3 lab / pos |
| --- | --- | --- | --- | --- | --- | --- |
| holdout all | 1,073 | 72 | 523 | 48.7% | 391 / 122 | 235 / 14 |
| holdout dark | 491 | 32 | 0 | 0.0% | 0 / 0 | 132 / 7 |
| holdout ERP | 582 | 40 | 523 | 89.9% | 391 / 122 | 103 / 7 |


## 12. Drop 12 chronic Y2 names — Y3 only

Chronic 12 Y2 names (0158/0172, ≥50% labeled months below 0): 12. Y3-only CN 0.579 → drop-12 0.579 (does not flip). Days stays 0.708. No Y2 AUROC claim.

| slice | CN | days | size |
| --- | --- | --- | --- |
| Y3 all | 0.579 | 0.711 | 0.617 |
| Y3 drop-12 | 0.579 | 0.708 | 0.617 |
| Y7 all (robustness) | 0.545 | — | — |
| Y7 drop-12 (robustness) | 0.545 | — | — |


## 13. Quintiles and CN>0 vs CN=0 rates

Y7 rate CN=0 25.1% vs CN>0 30.7%. A leftover Q4 would show a monotone CN quintile. A twin of thin issuance would pile with low issued.

| y | CN q | n | n_pos | rate | CN p50 |
| --- | --- | --- | --- | --- | --- |
| y7_top1_lost | Q1 | 1,462 | 341 | 23.3% | 0.000 |
| y7_top1_lost | Q2 | 1,461 | 380 | 26.0% | 0.000 |
| y7_top1_lost | Q3 | 1,462 | 359 | 24.6% | 0.000 |
| y7_top1_lost | Q4 | 1,461 | 369 | 25.3% | 0.016 |
| y7_top1_lost | Q5 | 1,462 | 573 | 39.2% | 0.161 |
| y3_recover_cash_6m | Q1 | 616 | 40 | 6.5% | 0.000 |
| y3_recover_cash_6m | Q2 | 616 | 48 | 7.8% | 0.000 |
| y3_recover_cash_6m | Q3 | 615 | 40 | 6.5% | 0.000 |
| y3_recover_cash_6m | Q4 | 616 | 26 | 4.2% | 0.013 |
| y3_recover_cash_6m | Q5 | 616 | 23 | 3.7% | 0.154 |


| y | slice | n_cm | n_lab | n_pos | rate |
| --- | --- | --- | --- | --- | --- |
| y7_top1_lost | CN=0 | 6,470 | 3,977 | 999 | 25.1% |
| y7_top1_lost | CN>0 | 4,737 | 3,331 | 1,023 | 30.7% |
| y7_top1_lost | CN NaN | 9,950 | 156 | 127 | 81.4% |
| y3_recover_cash_6m | CN=0 | 6,470 | 1,724 | 123 | 7.1% |
| y3_recover_cash_6m | CN>0 | 4,737 | 1,355 | 54 | 4.0% |
| y3_recover_cash_6m | CN NaN | 9,950 | 2,569 | 225 | 8.8% |


Plot: `credit_note_quintiles.png`.

## 14. Nonzero-only leftover

Y7 CN among CN>0 0.593 / among CN=0 0.500; residual after issued_lag1 on CN>0 0.594. Nonzero CN still measured.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | defined | 7,308 | 2,022 | 0.545 | 0.041 | 1 | 0.505 0.546 0.552 0.513 0.608 |
| y7_top1_lost | CN>0 | 3,331 | 1,023 | 0.593 | 0.070 | 1 | 0.558 0.498 0.650 0.589 0.669 |
| y7_top1_lost | CN=0 | 3,977 | 999 | 0.500 | 0.000 | 1 | 0.500 0.500 0.500 0.500 0.500 |
| y7_top1_lost | resid after issued_lag1 | CN>0 | 3,257 | 999 | 0.594 | 0.069 | 1 | 0.564 0.499 0.652 0.585 0.669 |
| y3_recover_cash_6m | defined | 3,079 | 177 | 0.579 | 0.065 | -1 | 0.655 0.509 0.518 0.627 0.584 |
| y3_recover_cash_6m | CN>0 | 1,355 | 54 | 0.406 | 0.067 | 1 | — 0.482 0.324 0.386 0.433 |
| y3_recover_cash_6m | CN=0 | 1,724 | 123 | 0.500 | 0.000 | 1 | 0.500 0.500 0.500 0.500 0.500 |
| y3_recover_cash_6m | resid after issued_lag1 | CN>0 | 1,355 | 54 | 0.405 | 0.072 | 1 | — 0.482 0.311 0.394 0.432 |


## 15. Company-mean leftover after company-mean issued_lag1

Y7 company-mean CN leftover after company-mean issued_lag1 0.577 (R²=0.009). Company-mean leftover still lives.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | company-mean CN | 7,464 | 2,149 | 0.566 | 0.074 | 1 | 0.541 0.463 0.565 0.596 0.664 |
| y7_top1_lost | mean CN after mean issued_lag1 | 7,464 | 2,149 | 0.577 | 0.076 | 1 | 0.549 0.462 0.612 0.598 0.665 |
| y3_recover_cash_6m | company-mean CN | 3,618 | 264 | 0.490 | 0.035 | -1 | 0.519 0.527 0.467 0.442 0.495 |
| y3_recover_cash_6m | mean CN after mean issued_lag1 | 3,618 | 264 | 0.475 | 0.057 | 1 | 0.541 0.481 0.394 0.446 0.512 |


## 16. CN lag1 leftover after issued_lag1

Y7 CN_lag1 leftover after issued_lag1 0.599 (R²=0.002). Lag leftover lives.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | CN_lag1 resid after issued_lag1 | 7,071 | 1,969 | 0.599 | 0.034 | 1 | 0.600 0.574 0.626 0.557 0.638 |
| y3_recover_cash_6m | CN_lag1 resid after issued_lag1 | 3,063 | 181 | 0.510 | 0.015 | 1 | 0.510 0.514 0.527 0.485 0.511 |


## 17. Leftover artifact — same-n raw vs residual

Same-n Y7: raw CN 0.548 vs OLS leftover 0.597 (Δ 0.049) vs rank leftover 0.609 vs log leftover 0.621; issued_lag1 0.620. Leftover lifts raw CN on the same rows — residual is not a sample trick.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | raw CN | issued_lag1 nn | 7,105 | 1,951 | 0.548 | 0.042 | 1 | 0.506 0.553 0.557 0.513 0.611 |
| y7_top1_lost | issued_lag1 | same n | 7,105 | 1,951 | 0.620 | 0.037 | -1 | 0.636 0.655 0.570 0.594 0.646 |
| y7_top1_lost | OLS resid CN~issued_lag1 | 7,105 | 1,951 | 0.597 | 0.038 | 1 | 0.607 0.575 0.631 0.542 0.631 |
| y7_top1_lost | rank resid CN~issued_lag1 | 7,105 | 1,951 | 0.609 | 0.036 | 1 | 0.613 0.610 0.631 0.549 0.643 |
| y7_top1_lost | OLS resid CN~log1p issued_lag1 | 7,105 | 1,951 | 0.621 | 0.037 | 1 | 0.619 0.606 0.652 0.570 0.661 |
| y7_top1_lost | raw CN all labeled nn | 7,308 | 2,022 | 0.545 | 0.041 | 1 | 0.505 0.546 0.552 0.513 0.608 |


## 18. Dark exception COMP_0962 (refund-only, not a 0-fill)

Dark CN non-null rows: 1. COMP_0962 is refund-only (no book invoice) — CN=1 is defined, not a 0-fill. The other 469 dark companies stay NaN. Do not fill 0.

| company_id | types | n_docs | has_invoice | CN months | CN values |
| --- | --- | --- | --- | --- | --- |
| COMP_0962 | refund | 1 | no | 1 | 1 |


## 19. Note vs refund ratios (in-memory, not a Y)

Y7 note_ratio 0.541 refund_ratio 0.496; note leftover after issued_lag1 0.595. If leftover is note not refund, the stand-in book is the object — still not a Y.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | note_ratio | 7,308 | 2,022 | 0.541 | 0.041 | 1 | 0.503 0.544 0.551 0.506 0.603 |
| y7_top1_lost | refund_ratio | 7,300 | 2,016 | 0.496 | 0.002 | -1 | 0.498 0.498 0.497 0.492 0.497 |
| y7_top1_lost | note_ratio resid issued_lag1 | 7,105 | 1,951 | 0.595 | 0.041 | 1 | 0.605 0.576 0.632 0.534 0.627 |
| y3_recover_cash_6m | note_ratio | 3,079 | 177 | 0.580 | 0.063 | -1 | 0.644 0.508 0.518 0.624 0.604 |
| y3_recover_cash_6m | refund_ratio | 3,075 | 176 | 0.494 | 0.019 | -1 | 0.517 0.486 0.504 0.466 0.496 |
| y3_recover_cash_6m | note_ratio resid issued_lag1 | 3,079 | 177 | 0.490 | 0.058 | -1 | 0.525 0.469 0.475 0.414 0.567 |


## 20. Fold 4 leftover vs issued / issued-CV

Fold 4 leftover check: CN 0.608 issued_lag1 0.647 issued-CV 0.605 CN residual 0.631. TURNOVER 0.680. Issued + CV own fold 4; CN residual does not save it.

| feature | n_va | n_pos | fold4 |
| --- | --- | --- | --- |
| CN | 1,768 | 668 | 0.608 |
| issued_lag1 | 1,734 | 657 | 0.647 |
| issued_lag_cv | 1,734 | 657 | 0.605 |
| CN resid issued_lag1 | 1,707 | 639 | 0.631 |
| DSO | 1,553 | 528 | 0.342 |


## 21. Within-company leftover (demean after demean issued)

Y7 demean CN 0.508; demean leftover after demean issued_lag1 0.509 (R²=0.000). Within-company leftover dies — KEEP leftover is who-uses-notes, not this month’s correction.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | demean CN | 7,308 | 2,022 | 0.508 | 0.052 | -1 | 0.531 0.423 0.525 0.561 0.501 |
| y7_top1_lost | demean CN after demean issued_lag1 | 7,105 | 1,951 | 0.509 | 0.056 | -1 | 0.542 0.418 0.522 0.562 0.502 |
| y3_recover_cash_6m | demean CN | 3,079 | 177 | 0.548 | 0.027 | -1 | 0.570 0.506 0.571 0.546 0.546 |
| y3_recover_cash_6m | demean CN after demean issued_lag1 | 3,079 | 177 | 0.543 | 0.027 | -1 | 0.573 0.500 0.552 0.548 0.543 |


## 22. Leftover inside issued_lag1 company terciles

Y7 leftover after issued_lag1 inside company issued terciles that stay ≥0.55: ['T2', 'T3']. If leftover dies inside every tercile it is still a thin-issuance mix.

| issued tercile | n | n_pos | CN | resid | issued_lag1 |
| --- | --- | --- | --- | --- | --- |
| T1 | 1,350 | 638 | 0.496 | 0.531 | 0.620 |
| T2 | 3,045 | 772 | 0.599 | 0.654 | 0.575 |
| T3 | 2,913 | 612 | 0.530 | 0.579 | 0.587 |


## 23. Leftover fold bits

Leftover folds [0.607 0.575 0.631 0.542 0.631] vs issued_lag1 [0.643 0.662 0.590 0.605 0.647]. If leftover fold 4 is the only lift, do not KEEP on that fold.

| feature | CV | folds | fold4 |
| --- | --- | --- | --- |
| CN | 0.545 | 0.505 0.546 0.552 0.513 0.608 | 0.608 |
| issued_lag1 | 0.630 | 0.643 0.662 0.590 0.605 0.647 | 0.647 |
| resid after issued_lag1 | 0.597 | 0.607 0.575 0.631 0.542 0.631 | 0.631 |
| resid after DSO | 0.479 | 0.465 0.556 0.506 0.509 0.357 | 0.357 |


## What failed / next (held for wave note)

- dark CN nn=1 is refund-only COMP_0962 — not a 0-fill; 469 others stay NaN
- Y3 CN 0.579 leftover-after-days 0.560 — DROP from the 44
- Q6 CN lag1 short 0.542 CLOSE (q6_quoted 0.542)

Elapsed 4s. Cuts: prevalence, formula, twins, singles, Y7 leftover after issued_lag1/DSO, Y3 leftover after days, token mix, Q6, fold 4, ICC, dark/holdout, 12 Y2 names (Y3 only), quintiles, nonzero pile, company-mean, lag leftover, same-n artifact, COMP_0962, note vs refund, fold-4 leftover, demean leftover, issued terciles, leftover folds.

## What this module did not do

- Did not change night Y3 0.762 / 0.752 or Y7 TURNOVER 0.720 / B_shallow 0.712.
- Did not put CN on the Y3 15-col card. Did not grow TURNOVER. Did not use Family D as Y7 X.
- Did not invent `y_credit_note`. Did not write 0–100 / pillars. Did not touch `product/`.
- Did not edit `invoices.py` / `gbm_y7_core.py` / sibling `zero_in_qa` / `growth_qa` / `recency_qa`.
- Did not rewrite parquet or duckdb. Did not run `build_targets`. Did not commit.
- Did not quote holdout Y7 AUROC (122 pos is coverage only).
