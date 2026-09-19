# Unused leftover of AR issued to last month's top-1 after issued_lag1

Generated `2026-09-19T07:54:46+02:00` by agent `689100e7`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_cust_lost`. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Do **not** grow TURNOVER. Y7 never D as engine X. Y5 never E. Y3 never B. Dark 470 stay NaN not 0.

`e_issued_top1` = this-month AR |amt| issued to **last month’s** trailing-3m top-1 counterparty (Y7 object, lagged). 0 if that buyer exists and got nothing; NaN if no lag-1 top-1. Jacobson demand-shrinkage / Irvine major-customer / Amberg issued −1 pp. Not `d_cust_lost` (Y3 leftover 0.522 DROP).

## Headline

e_issued_top1 defined 40.6% of train CM; dark 470 NaN CONFIRM. zero-among-defined 39.9%. ρ vs issued_lag1 0.456 issued 0.615 d_cust_top1 0.196 not a twin. Y7 raw 0.810 vs issued_lag1 0.630 size 0.469 beat-size PASS. Leftover after issued_lag1 rank 0.789 OLS 0.779 (after contemp issued 0.785; share 0.812). ICC 0.863 demean leftover 0.548. Boot p05 0.774. Y7 leftover **KEEP**. Engine is thinning-to-zero (Y7 rate 57.7% vs 8.1% when named buyer still billed); positive-only leftover 0.576 < 0.58. CLOSE TURNOVER add-on. issued_lag1 Q6 0.626 stays.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_cust_lost`. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Thinning is a *why*, not a recovery clock. |
| 3 | Who is turning? | Named-buyer issued going to 0 is a turn *if* leftover lives. |
| 4 | Dip vs fall? | Y7 leftover after issued_lag1 **KEEP** — leftover after issued_lag1 0.789 ≥ 0.58, not a twin, beat-size PASS |
| 5 | Why did it change? | Identity of the missing euro. Firm issued_lag1 cannot see *who* thinned. |
| 6 | Months earlier? | **KEEP** named-volume lead footnote — Q6 short e_issued_top1 0.804 lag1 0.759 vs issued_lag1 0.626 (quote 0.626). KEEP this lag1. issued_lag1 0.626 stays on TURNOVER. |

## PARK / CLOSE / KEEP / DROP

| object | decision | why |
| --- | --- | --- |
| issued-to-last-month-top-1 leftover after issued_lag1 (Y7) | **KEEP** | leftover after issued_lag1 0.789 ≥ 0.58, not a twin, beat-size PASS |
| TURNOVER add-on | **CLOSE** | do not grow 0.720 |
| D as Y7 engine X | **DROP** | Y7 never D |
| as Y3 X / the 15-col card | **DROP** | leftover after days 0.571 — off the 15-col card; d_cust_lost 0.522 already DROP |
| as Y5 X | **DROP** | Y5 never E |
| health Y `y_cust_lost` | **PARK** | `d_cust_lost` leftover 0.522 already DROP |
| Q6 named-volume lead (not a TURNOVER add-on) | **KEEP** | leftover / lag1 ≥0.58; issued_lag1 0.626 stays the card lead |
| issued_lag1 Q6 0.626 | **KEEP (locked)** | do not overwrite issued_qa |

## 1. Coverage / dark 470

Train CM=21,157 / 1,214 companies. e_issued_top1 defined 8,594 (40.6%) zero-among-defined 39.9%. Dark 470 nn=0 zero=0 CONFIRM NaN not 0.

| item | value |
| --- | ---: |
| train CM / companies | 21,157 / 1,214 |
| e_issued_top1 defined | 8,594 (40.6%) |
| among defined: zero | 39.9% |
| mean / p50 issued-to-top1 | 1495850.869 / 6291.120 |
| mean share of this-month issued | 0.424 |
| dark 470 nn / zero | 0 / 0 |
| dark NaN | CONFIRM |
| Y7 labeled / pos / issued-top1-nn | 7,464 / 2,149 / 6,721 |
| acf1 / acf3 | 0.035 / -0.058 |

## 2. Formula (0 if named and silent; NaN if no lag-1 top-1)

Named lag-1 top-1 rows 8,594; issued-to-them defined 8,594 (zero 3,431). Unnamed nn=0 MATCH — 0 only when named, NaN otherwise.

## 3. Spearman twins (|ρ|≥0.80)

Gate twins vs issued_lag1 / issued / d_cust_top1: none. ρ vs issued_lag1=0.456 issued=0.615 top1=0.196 lost=-0.148 size=0.262.

| vs | ρ | n | twin? |
| --- | --- | --- | --- |
| e_ar_issued_lag1 | 0.456 | 8,594 |  |
| e_ar_issued | 0.615 | 8,594 |  |
| d_cust_top1 | 0.196 | 8,125 |  |
| d_cust_lost | -0.148 | 8,310 |  |
| log_in3 | 0.262 | 8,383 |  |
| e_issued_top1_share | 0.711 | 7,464 |  |
| top1_share_lag1 | 0.110 | 8,594 |  |


## 4. Single-feature train group-fold AUROC

Night issued_lag1 **0.630** (replica 0.630). Days **0.711** (replica 0.711). Size **0.617**. TURNOVER **0.720** / 0.712 unchanged.

Y7 e_issued_top1 0.810 vs issued_lag1 0.630 (night 0.630) size 0.469 beat-size PASS Δ=0.341. Y3 e_issued_top1 0.624 vs days 0.711.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | iss_top1_y7 | 6,721 | 1,869 | 0.810 | 0.039 | -1 | 0.808 0.869 0.764 0.792 0.816 |
| y7_top1_lost | share_y7 | 5,957 | 1,375 | 0.808 | 0.023 | -1 | 0.825 0.822 0.770 0.801 0.822 |
| y7_top1_lost | iss_lag1_y7 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y7_top1_lost | iss_y7 | 7,464 | 2,149 | 0.663 | 0.037 | -1 | 0.660 0.702 0.615 0.641 0.696 |
| y7_top1_lost | size_y7 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 |
| y7_top1_lost | lost_y7 | 6,849 | 1,930 | 0.603 | 0.050 | 1 | 0.604 0.597 0.527 0.622 0.666 |
| y3_recover_cash_6m | iss_top1_y3 | 2,556 | 132 | 0.624 | 0.060 | -1 | 0.709 0.575 0.559 0.629 0.650 |
| y3_recover_cash_6m | days_y3 | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | size_y3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |


## 5. Residual Y7 after issued_lag1 (KEEP gate)

KEEP leftover only if rank ≥ **0.58**, not a twin of issued_lag1 / issued / `d_cust_top1`, not SIZE, beat-size. Rank leftover is honest; OLS can fake.

Y7 leftover after issued_lag1 rank 0.789 OLS 0.779 ρ(resid,issued_lag1)=0.174 R²=0.064 KEEP ≥0.58. After contemp issued 0.785; share leftover 0.812.

| cut | rank | OLS | ρ(resid,ctrl) | R² | n | dies? |
| --- | --- | --- | --- | --- | --- | --- |
| after issued_lag1 | 0.789 | 0.779 | 0.174 | 0.064 | 6,721 | lives |
| after e_ar_issued | 0.785 | 0.776 | 0.287 | 0.162 | 6,721 | lives |
| after issued_lag1+issued | 0.783 | 0.768 | 0.155 | 0.173 | 6,721 | lives |
| after size | 0.808 | 0.451 | -0.912 | 0.011 | 6,517 | dies |
| after CN | 0.800 | 0.704 | -0.665 | 0.002 | 6,576 | lives |
| after delay | 0.807 | 0.627 | -0.898 | 0.006 | 4,992 | dies |
| after d_cust_top1 | 0.804 | 0.662 | 0.869 | 0.000 | 6,262 | dies |
| share after issued_lag1 | 0.812 | 0.810 | 0.094 | 0.002 | 5,957 | lives |


## 6. Residual Y3 after days (off the card)

Y3 leftover after days rank 0.571 OLS 0.728 dies — stay off the 15-col card.

| cut | rank | OLS | dies? |
| --- | --- | --- | --- |
| Y3 leftover after days | 0.571 | 0.728 | dies |


## 7. Q6 — lag1 on short vs long

Q6 short e_issued_top1 0.804 lag1 0.759 vs issued_lag1 0.626 (quote 0.626). KEEP this lag1.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | all e_issued_top1 | 6,721 | 1,869 | 0.810 | 0.039 | -1 | 0.808 0.869 0.764 0.792 0.816 |
| y7_top1_lost | all e_issued_top1_lag1 | 6,038 | 1,620 | 0.761 | 0.045 | -1 | 0.761 0.825 0.706 0.734 0.780 |
| y7_top1_lost | all issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y7_top1_lost | short_<12 e_issued_top1 | 3,765 | 1,103 | 0.804 | 0.032 | -1 | 0.805 0.851 0.774 0.773 0.818 |
| y7_top1_lost | short_<12 e_issued_top1_lag1 | 3,149 | 880 | 0.759 | 0.054 | -1 | 0.772 0.831 0.709 0.701 0.781 |
| y7_top1_lost | short_<12 issued_lag1 | 4,210 | 1,260 | 0.626 | 0.068 | -1 | 0.650 0.703 0.615 0.519 0.645 |
| y7_top1_lost | long_>=18 e_issued_top1 | 895 | 197 | 0.853 | 0.061 | -1 | 0.865 0.938 0.780 0.871 0.809 |
| y7_top1_lost | long_>=18 e_issued_top1_lag1 | 880 | 189 | 0.810 | 0.071 | -1 | 0.862 0.903 0.734 0.797 0.756 |
| y7_top1_lost | long_>=18 issued_lag1 | 916 | 209 | 0.646 | 0.077 | -1 | 0.631 0.601 0.552 0.742 0.705 |


## 8. ICC / demean

ICC 0.863 (k=670) BETWEEN / who-issues-to-whom style. Demean leftover after issued_lag1 0.548 dies — who-has-a-big-buyer.

| cut | value | k |
| --- | --- | --- |
| ICC | 0.863 | 670 |
| company-mean leftover Y7 | 0.737 | — |
| demean leftover after issued_lag1 | 0.548 | — |


## 9. Dark 470 + holdout coverage (no AUROC)

Holdout 72 coverage only — no AUROC. Holdout e_issued_top1 nn=463. Holdout dark nn=0 (want 0).

| slice | n_co | nn | zero |
| --- | --- | --- | --- |
| train dark | 470 | 0 | 0 |
| holdout (coverage only) | 72 | 463 | 196 |
| holdout dark | 32 | 0 | 0 |


## 10. Quintiles (thinning seat)

Y7 rate by rank-quintile of issued-to-last-month-top-1. Q5−Q1 -52.3% (Jacobson/Irvine thinning seat; sign expected negative if thinning → loss).

| q | n | rate | p50 issued-to-top1 |
| --- | --- | --- | --- |
| Q1 | 1,345 | 58.7% | 0.000 |
| Q2 | 1,344 | 56.2% | 0.000 |
| Q3 | 1,344 | 11.8% | 6113.320 |
| Q4 | 1,344 | 5.7% | 54067.815 |
| Q5 | 1,344 | 6.5% | 387710.795 |


Zero vs positive issued-to-top-1 on the same labeled rows:

| bin | n | share | Y7 rate |
| --- | --- | --- | --- |
| positive | 4,049 | 60.2% | 8.1% |
| zero | 2,672 | 39.8% | 57.7% |


Plot: `issued_top1_qa.png`.

## 11. Company bootstrap leftover after issued_lag1

Bootstrap leftover-after-issued_lag1 rank p05=0.774 p50=0.792 p95=0.805 share<0.55=0.0% n=40.

## 12. Extra — monopoly vs diversified (last-month top-1 share)

Diversified leftover 0.755; monopoly leftover 0.812; HHI-tail leftover 0.828.

| cut | rank | OLS | n | n_pos | ρ vs issued_lag1 | dies? |
| --- | --- | --- | --- | --- | --- | --- |
| share_lag1 <0.50 diversified | 0.755 | 0.713 | 2,585 | 877 | 0.319 | lives ≥0.58 |
| share_lag1 0.50–0.80 | 0.787 | 0.794 | 1,478 | 346 | 0.517 | lives ≥0.58 |
| share_lag1 ≥0.80 monopoly | 0.812 | 0.820 | 2,658 | 646 | 0.589 | lives ≥0.58 |
| share_lag1 >0.975 HHI-tail seat | 0.828 | 0.705 | 1,611 | 408 | 0.631 | dies |
| zero issued-to-top1 | 0.571 | 0.500 | 2,672 | 1,541 | — | lives |
| positive issued-to-top1 | 0.576 | 0.592 | 4,049 | 328 | 0.735 | lives |


## 13. Fold 4 + same-n artifact

Fold 4 e_issued_top1 0.816 vs issued_lag1 0.647. Same-n raw 0.810 vs leftover 0.789 lift 0.020. TURNOVER fold-4 0.680 stays issued. Do not grow the card.

| fold | e_issued_top1 | issued_lag1 |
| --- | --- | --- |
| 4 | 0.816 | 0.647 |


## 14. Extra — same vs switched top-1; zero dummy; Q6 leftover

Same-id rows 5,230 leftover 0.816; switched 1,491 leftover 0.611. Y7 rate zero=57.7% vs positive=8.1%. Zero dummy leftover 0.774; positive-only leftover 0.576; Q6 leftover of lag1 after issued_lag1 0.749.

| cut | rank | OLS | n | n_pos | ρ vs issued_lag1 | dies? |
| --- | --- | --- | --- | --- | --- | --- |
| same top-1 id (lag1==t) | 0.816 | 0.811 | 5,230 | 1,210 | 0.465 | lives ≥0.58 |
| switched top-1 id | 0.611 | 0.605 | 1,491 | 659 | 0.249 | lives ≥0.58 |
| positive issued-to-top1 only | 0.576 | 0.592 | 4,049 | 328 | 0.735 | lives |
| zero issued-to-top1 only | 0.571 | 0.500 | 2,672 | 1,541 | — | lives |
| zero dummy after issued_lag1 | 0.774 | 0.818 | 6,721 | 1,869 | -0.696 | lives |
| issued_top1_lag1 after issued_lag1 (Q6 leftover) | 0.749 | 0.743 | 6,038 | 1,620 | 0.284 | lives |


## What this note did not do

- Did not change Y7 0.720 / 0.712 or Y3 0.762 / 0.752.
- Did not put issued-to-top-1 / delay / CN / DSO on TURNOVER.
- Did not merge Family D as Y7 X. Did not score this as Y5 X.
- Did not overwrite delay_qa / issued_qa / credit_note_qa / top1_pastdue_qa / top1_qa / cust_lost_qa / y7_core.
- Did not invent `y_cust_lost`. Dark 470 stayed NaN.
