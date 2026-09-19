# Unused leftover of top-1 AR PastDue% after issued_lag1 + CN

Generated `2026-09-19T07:44:34+02:00` by agent `689100e7`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_delay`. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Do **not** grow TURNOVER. Y7 never D as engine X. Y5 never E. Y3 never B. Dark 470 stay NaN not 0.

`e_top1_pastdue` = overdue |amt| / open |amt| of the trailing-3m top-1 AR counterparty (same object as Y7) at period end. `e_top1_pastdue_30` is the BdF >30d cut. Hirshleifer: buyer PastDue% on the supplier tape. Not firm-level `e_delay_coll`.

## Headline

top-1 PastDue defined 32.6% of train CM; dark 470 NaN CONFIRM. mean 0.586 (Hirshleifer 27%). ρ vs delay 0.278 DSO 0.291 overdue 0.858 TWIN. Y7 raw 0.612 vs issued_lag1 0.630 size 0.469 beat-size PASS. Leftover after issued_lag1+CN rank 0.576 OLS 0.621 (issued-only 0.567; after delay 0.597). ICC 0.918 demean leftover 0.569. Boot p05 0.551. Y7 leftover **CLOSE**. Do not grow TURNOVER 0.720. Delay footnote 0.581 stays.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_delay`. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Not a recovery clock. |
| 3 | Who is turning? | Top-1 already late is a *why*, not a cash turn. |
| 4 | Dip vs fall? | Y7 leftover after issued_lag1+CN **CLOSE** — leftover 0.576 dies / twin=['e_ar_overdue'] — keep the delay footnote 0.581 |
| 5 | Why did it change? | Hirshleifer seat: *which* customer is late. Firm delay leftover 0.581 stays the footnote if this dies. |
| 6 | Months earlier? | **KEEP** — Q6 short e_top1_pastdue 0.613 lag1 0.581 vs issued_lag1 0.626 (quote 0.626). KEEP lag1. |

## PARK / CLOSE / KEEP / DROP

| object | decision | why |
| --- | --- | --- |
| top-1 PastDue leftover after issued_lag1+CN (Y7) | **CLOSE** | leftover 0.576 dies / twin=['e_ar_overdue'] — keep the delay footnote 0.581 |
| TURNOVER add-on / delay / CN / DSO back on the card | **CLOSE** | do not grow 0.720 |
| D as Y7 engine X | **DROP** | Y7 never D |
| as Y3 X / the 15-col card | **DROP** | leftover after days 0.451 — off the 15-col card |
| as Y5 X | **DROP** | Y5 never E |
| health Y `y_delay` / `y_top1_pastdue` | **PARK** | do not invent it |
| Q6 lag1 | **KEEP** | issued_lag1 0.626 stays the invoice lead unless lag1 ≥0.58 |
| delay leftover 0.581 footnote | **KEEP (locked)** | do not overwrite delay_qa |

## 1. Coverage / dark 470

Train CM=21,157 / 1,214 companies. e_top1_pastdue defined 6,906 (32.6%) zero-among-defined 28.0%. Dark 470 nn=0 zero=0 CONFIRM NaN not 0.

| item | value |
| --- | ---: |
| train CM / companies | 21,157 / 1,214 |
| e_top1_pastdue defined | 6,906 (32.6%) |
| among defined: zero | 28.0% |
| mean / p50 PastDue | 0.586 / 0.798 |
| mean PastDue30 | 0.401 |
| dark 470 nn / zero | 0 / 0 |
| dark NaN | CONFIRM |
| Y7 labeled / pos / PastDue-nn | 7,464 / 2,149 / 5,544 |
| acf1 / acf3 | 0.242 / -0.027 |

## 2. Formula (overdue / open of named top-1)

Recompute overdue/open vs e_top1_pastdue: n_both=6,906 max|Δ|=0.00e+00 exact=6,906 MATCH.

## 3. Spearman twins (|ρ|≥0.80)

Gate twins vs delay/DSO/overdue/CN: ['e_ar_overdue']. ρ vs delay_coll=0.278 DSO=0.291 ar_overdue=0.858 CN=-0.097 issued_lag1=-0.095 top1=0.077 size=-0.101.

| vs | ρ | n | twin? |
| --- | --- | --- | --- |
| e_delay_coll | 0.278 | 4,711 |  |
| e_dso_proxy | 0.291 | 6,296 |  |
| e_ar_overdue | 0.858 | 6,906 | TWIN |
| e_ar_overdue_30 | 0.601 | 6,906 |  |
| e_credit_note_ratio | -0.097 | 6,760 |  |
| e_ar_issued_lag1 | -0.095 | 6,735 |  |
| d_cust_top1 | 0.077 | 6,372 |  |
| log_in3 | -0.101 | 6,517 |  |
| e_top1_pastdue_30 | 0.683 | 6,906 |  |


## 4. Single-feature train group-fold AUROC

Night issued_lag1 **0.630** (replica 0.630). Days **0.711** (replica 0.711). Size **0.617**. TURNOVER **0.720** / 0.712 unchanged.

Y7 e_top1_pastdue 0.612 vs issued_lag1 0.630 (night 0.630) CN 0.545 delay 0.574 size 0.469 beat-size PASS Δ=0.144. Y3 e_top1_pastdue 0.545 vs days 0.711.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | pd_y7 | 5,544 | 1,356 | 0.612 | 0.036 | 1 | 0.608 0.642 0.551 0.631 0.629 |
| y7_top1_lost | pd30_y7 | 5,544 | 1,356 | 0.552 | 0.033 | 1 | 0.573 0.598 0.520 0.541 0.527 |
| y7_top1_lost | iss_y7 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y7_top1_lost | cn_y7 | 7,308 | 2,022 | 0.545 | 0.041 | 1 | 0.505 0.546 0.552 0.513 0.608 |
| y7_top1_lost | delay_y7 | 5,158 | 1,304 | 0.574 | 0.041 | 1 | 0.569 0.572 0.518 0.633 0.576 |
| y7_top1_lost | dso_y7 | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 |
| y7_top1_lost | size_y7 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 |
| y3_recover_cash_6m | pd_y3 | 1,965 | 97 | 0.545 | 0.060 | 1 | 0.543 0.477 0.533 0.642 0.531 |
| y3_recover_cash_6m | days_y3 | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | size_y3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |


## 5. Residual Y7 after issued_lag1 + CN (and companions)

KEEP leftover only if rank ≥ **0.58**, not a twin, not SIZE, beat-size. Rank leftover is honest; OLS can fake.

Y7 leftover after issued_lag1+CN rank 0.576 OLS 0.621 ρ(resid,issued)=-0.127 R²=0.001 lives but <0.58. After issued only 0.567; after delay 0.597; after DSO 0.538.

| cut | rank | OLS | ρ(resid,ctrl) | R² | n | dies? |
| --- | --- | --- | --- | --- | --- | --- |
| after issued_lag1+CN | 0.576 | 0.621 | -0.127 | 0.001 | 5,259 | lives |
| after issued_lag1 | 0.567 | 0.652 | -0.310 | 0.000 | 5,373 | lives |
| after CN | 0.605 | 0.604 | 0.112 | 0.000 | 5,422 | lives |
| after delay | 0.597 | 0.573 | -0.057 | 0.054 | 3,601 | lives |
| after DSO | 0.538 | 0.592 | 0.474 | 0.000 | 5,024 | dies |
| after ar_overdue | 0.518 | 0.484 | -0.012 | 0.728 | 5,544 | dies |
| after size | 0.607 | 0.595 | 0.191 | 0.008 | 5,155 | lives |


## 6. Residual Y3 after days (off the card)

Y3 leftover after days rank 0.451 OLS 0.452 dies — stay off the 15-col card.

| cut | rank | OLS | dies? |
| --- | --- | --- | --- |
| Y3 leftover after days | 0.451 | 0.452 | dies |


## 7. Q6 — lag1 on short vs long

Q6 short e_top1_pastdue 0.613 lag1 0.581 vs issued_lag1 0.626 (quote 0.626). KEEP lag1.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | all e_top1_pastdue | 5,544 | 1,356 | 0.612 | 0.036 | 1 | 0.608 0.642 0.551 0.631 0.629 |
| y7_top1_lost | all e_top1_pastdue_lag1 | 4,979 | 1,204 | 0.579 | 0.020 | 1 | 0.578 0.582 0.547 0.601 0.587 |
| y7_top1_lost | all issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y7_top1_lost | short_<12 e_top1_pastdue | 3,300 | 886 | 0.613 | 0.026 | 1 | 0.605 0.630 0.575 0.610 0.643 |
| y7_top1_lost | short_<12 e_top1_pastdue_lag1 | 2,794 | 740 | 0.581 | 0.023 | 1 | 0.602 0.578 0.553 0.566 0.606 |
| y7_top1_lost | short_<12 issued_lag1 | 4,210 | 1,260 | 0.626 | 0.068 | -1 | 0.650 0.703 0.615 0.519 0.645 |
| y7_top1_lost | long_>=18 e_top1_pastdue | 701 | 132 | 0.679 | 0.054 | 1 | 0.677 0.680 0.631 0.766 0.639 |
| y7_top1_lost | long_>=18 e_top1_pastdue_lag1 | 673 | 127 | 0.654 | 0.086 | 1 | 0.506 0.655 0.684 0.713 0.710 |
| y7_top1_lost | long_>=18 issued_lag1 | 916 | 209 | 0.646 | 0.077 | -1 | 0.631 0.601 0.552 0.742 0.705 |


## 8. ICC / demean

ICC 0.918 (k=646) BETWEEN / who-is-late style like delay 0.922. Demean leftover after issued+CN 0.569 lives as a month shock.

| cut | value | k |
| --- | --- | --- |
| ICC | 0.918 | 646 |
| company-mean leftover Y7 | 0.608 | — |
| demean leftover after issued+CN | 0.569 | — |


## 9. Dark 470 + holdout coverage (no AUROC)

Holdout 72 coverage only — no AUROC. Holdout e_top1_pastdue nn=358. Holdout dark nn=0 (want 0).

| slice | n_co | nn | zero |
| --- | --- | --- | --- |
| train dark | 470 | 0 | 0 |
| holdout (coverage only) | 72 | 358 | 92 |
| holdout dark | 32 | 0 | 0 |


## 10. Quintiles (Hirshleifer Q5 vs Q1 seat)

Y7 rate by rank-quintile of top-1 PastDue (qcut on ranks — raw qcut dies on 0/1 pile). Q5−Q1 16.9%. 0/1 pile on labeled nn: see extras.

| q | n | rate | p50 PastDue |
| --- | --- | --- | --- |
| Q1 | 1,109 | 22.0% | 0.000 |
| Q2 | 1,109 | 16.3% | 0.000 |
| Q3 | 1,108 | 10.4% | 0.760 |
| Q4 | 1,109 | 34.7% | 1.000 |
| Q5 | 1,109 | 38.9% | 1.000 |


0/1 pile on the same labeled rows (raw qcut dies here):

| bin | n | share | Y7 rate | p50 PastDue |
| --- | --- | --- | --- | --- |
| all_late_1 | 2,087 | 37.6% | 38.6% | 1.000 |
| current_0 | 1,669 | 30.1% | 21.5% | 0.000 |
| interior | 1,788 | 32.3% | 10.7% | 0.653 |


Plot: `top1_pastdue_qa.png`.

## 11. Company bootstrap leftover after issued+CN

Bootstrap leftover-after-issued+CN rank p05=0.551 p50=0.578 p95=0.614 share<0.55=5.0% n=40.

## 12. Fold 4

Fold 4 e_top1_pastdue 0.629 vs issued_lag1 0.647. TURNOVER fold-4 0.680 stays issued. Do not grow the card.

| fold | e_top1_pastdue | issued_lag1 |
| --- | --- | --- |
| 4 | 0.629 | 0.647 |


## 13. Same-n artifact

Same-n raw e_top1_pastdue 0.597 vs leftover rank 0.576 lift 0.021.

## 14. Extra — 0/1 pile + all-late dummy leftover

0/1 pile on Y7-labeled nn: current 30.1% all-late 37.6% interior 32.3%. All-late dummy leftover after issued+CN rank 0.576 lives.

| bin | n | share of labeled-nn | Y7 rate |
| --- | --- | --- | --- |
| current_0 | 1,669 | 30.1% | 21.5% |
| interior_(0,1) | 1,788 | 32.3% | 10.7% |
| all_late_1 | 2,087 | 37.6% | 38.6% |
| nonzero | 3,875 | 69.9% | 25.7% |


## 15. Extra — leftover by top-1 share (monopoly vs diversified)

Diversified leftover 0.615 ρ-overdue 0.784; monopoly leftover 0.519 ρ-overdue 0.939. Hirshleifer named-buyer leftover on diversified books lives ≥0.58 — still not a TURNOVER add-on.

| cut | rank | OLS | n | n_pos | ρ vs overdue | twin? | dies? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| share <0.50 diversified | 0.615 | 0.621 | 1,943 | 496 | 0.784 |  | lives ≥0.58 |
| share 0.50–0.80 | 0.539 | 0.595 | 1,208 | 241 | 0.825 | TWIN | dies |
| share ≥0.80 monopoly | 0.519 | 0.621 | 2,108 | 465 | 0.939 | TWIN | dies |
| |gap| vs overdue ≥0.20 | 0.481 | 0.405 | 1,109 | 266 | 0.193 |  | dies |


## 16. Extra — PastDue30 + leftover after issued / delay / overdue

PastDue30 leftover after issued+CN 0.556. Triple after issued+CN+overdue 0.524. Gap (named − firm overdue) 0.543. After issued only 0.567; after delay 0.597.

| cut | rank | OLS | ρ(resid,ctrl) | n | dies? |
| --- | --- | --- | --- | --- | --- |
| pd30 after issued+CN | 0.556 | 0.555 | -0.047 | 5,259 | lives |
| pd after issued+CN+overdue | 0.524 | 0.517 | -0.021 | 5,259 | dies |
| pd after issued+CN+delay | 0.586 | 0.570 | -0.039 | 3,502 | lives |
| gap(pd−overdue) after issued+CN | 0.543 | 0.545 | -0.039 | 5,259 | dies |
| pd after issued only (locked extra) | 0.567 | 0.652 | -0.310 | 5,373 | lives |
| pd after delay only (locked extra) | 0.597 | 0.573 | -0.057 | 3,601 | lives |


## 17. Extra — company bootstrap leftover after delay

Bootstrap leftover-after-delay rank p05=0.558 p50=0.601 p95=0.641 share<0.55=5.0% n=40.

## 18. Extra — leftover on nonzero / interior PastDue

Nonzero leftover 0.613; interior leftover 0.527 (Hirshleifer continuous object). A slice lives ≥0.58 — still not TURNOVER.

| cut | rank | OLS | n | n_pos | ρ vs overdue | twin? | dies? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nonzero PastDue | 0.613 | 0.701 | 3,660 | 863 | 0.785 |  | lives ≥0.58 |
| interior (0,1) | 0.527 | 0.518 | 1,736 | 177 | 0.797 |  | dies |
| all-late only | 0.530 | 0.497 | 1,924 | 686 | — |  | dies |


## 19. Extra — diversified leftover after overdue (is 0.615 a rewrite?)

Diversified leftover after issued+CN+overdue 0.454 dies — 0.615 was an overdue rewrite just under the twin bar. Div twins: none.

| cut | rank | OLS | n | n_pos | ρ vs overdue | twin? | dies? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| div after issued+CN | 0.615 | 0.621 | 1,943 | 496 | 0.784 |  | lives ≥0.58 |
| div after issued+CN+overdue | 0.454 | 0.547 | 1,943 | 496 | 0.784 |  | dies |
| div after overdue only | 0.450 | 0.553 | 2,020 | 536 | 0.784 |  | dies |
| div after delay | 0.625 | 0.611 | 1,462 | 357 | 0.784 |  | lives ≥0.58 |


Twin screen on share<0.50 only:

| vs | ρ on div | n | twin? |
| --- | --- | --- | --- |
| e_delay_coll | 0.302 | 1,462 |  |
| e_dso_proxy | 0.285 | 1,954 |  |
| e_ar_overdue | 0.784 | 2,020 |  |
| e_credit_note_ratio | -0.119 | 1,995 |  |
| log_in3 | -0.140 | 1,893 |  |


## What this note did not do

- Did not change Y7 0.720 / 0.712 or Y3 0.762 / 0.752.
- Did not put PastDue / delay / CN / DSO on TURNOVER.
- Did not merge Family D as Y7 X. Did not score this as Y5 X.
- Did not overwrite delay_qa / credit_note_qa / issued_qa / dso_qa / top1_qa / lit_invoice.md.
- Did not invent `y_delay`. Dark 470 stayed NaN.
