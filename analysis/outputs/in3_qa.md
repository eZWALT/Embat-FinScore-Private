# Unused leftover of `a_in3` after days (size control)

Generated `2026-09-19T05:38:01+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_in3`. Do not put `a_in3` on the 15-col card. Do not overwrite `salary_month_qa.*`. Do not grow TURNOVER.

`a_in3` = trailing-3m operational inflow. Size bar is always `log1p(a_in3)` (quoted 0.617). This cut asks leftover after `c_n_days_with_tx`.

## Headline

`log1p(a_in3)` on the 44: **CLOSE as unused leftover** (unused leftover after days: honest rank 0.521 dies (OLS 0.531 fake=False). DROP from the 44 as engine X. Size bar 0.617 stays the KEEP-as-X quote, not a card stem.). Y3 leftover after days OLS 0.531 rank 0.521 (dies, fake=False). Inverse days after size rank 0.667. Single size 0.617 vs days 0.711. 15-col card: no — do not put a_in3 on the 15-col card. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Size is the **control**, not a health Y. CLOSE as unused leftover. |
| 2 | Who is improving? | Q6 lag leftover after days_lag1. a_growth_12 already CLOSE. |
| 3 | Who is turning? | Not a new Y. Do not invent `y_in3`. |
| 4 | Dip vs fall? | Not a TURNOVER column. |
| 5 | Why did it change? | Leftover after days rank 0.521. Twin: ['a_op_in', 'log_opin', 'a_in6', 'a_in12', 'log_in6', 'log_in12']. |
| 6 | Months earlier? | lag1 leftover after days_lag1 0.527; lag3 0.555. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `log1p(a_in3)` as 44-col size control | **CLOSE as unused leftover** | unused leftover after days: honest rank 0.521 dies (OLS 0.531 fake=False). DROP from the 44 as engine X. Size bar 0.617 stays the KEEP-as-X quote, not a card stem. |
| `a_in3` as engine X / 15-col card | **no** | no — do not put a_in3 on the 15-col card |
| `a_op_in` / `a_in6` / `a_in12` besides size | **CLOSE** | twins of the size stem |

## 1 — Coverage; acf1; SIZE ρ

Train 1,214 co / 21,157 CM. a_in3 cov 88.5% acf1=0.655 vs a_op_in acf1=0.004 (CONFIRM persist > a_op_in). ρ log_in3 vs days 0.595 vs a_op_in 0.876 vs a_in6 0.930 vs a_in12 0.841. Twins: ['a_op_in', 'log_opin', 'a_in6', 'a_in12', 'log_in6', 'log_in12'].

| col | n_nn | cov | acf1 |
| --- | --- | --- | --- |
| a_in3 | 18,729 | 88.5% | 0.655 |
| a_op_in | 21,157 | 100.0% | 0.004 |
| log1p(a_in3) | 18,729 | 88.5% | 0.646 |


| vs | rho | twin |
| --- | --- | --- |
| c_n_days_with_tx | 0.595 | no |
| a_op_in | 0.876 | yes |
| log_opin | 0.877 | yes |
| a_in6 | 0.930 | yes |
| a_in12 | 0.841 | yes |
| log_in6 | 0.930 | yes |
| log_in12 | 0.842 | yes |
| log_in3 | 1.000 | self |


## 2 — Single-feature group-fold Y3

Y3 log1p(a_in3) 0.617 vs days 0.711 vs raw a_in3 0.617 vs a_op_in 0.676. Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Engine beat-size vs itself: False (it IS the size bar).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | a_in3 | 5,528 | 391 | 0.617 | 0.619 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | a_op_in | 5,648 | 402 | 0.676 | 0.685 | 0.050 | -1 | 0.609 0.683 0.737 0.647 0.705 |
| y3_recover_cash_6m | log1p(a_op_in) | 5,648 | 402 | 0.676 | 0.685 | 0.050 | -1 | 0.609 0.684 0.737 0.647 0.705 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log1p(a_in6) | 4,212 | 313 | 0.601 | 0.604 | 0.071 | -1 | 0.511 0.634 0.666 0.540 0.654 |
| y3_recover_cash_6m | log1p(a_in12) | 1,925 | 150 | 0.557 | 0.565 | 0.055 | -1 | 0.537 0.521 0.524 0.551 0.653 |
| y3_recover_cash_6m | c_salary_month | 5,648 | 402 | 0.671 | 0.672 | 0.050 | -1 | 0.611 0.659 0.745 0.689 0.652 |


## 3 — Honest leftover after days

log1p(a_in3) leftover after days OLS 0.531 rank 0.521 fake=False ρ(resid,days)=-0.163 R²=0.336 honest_dies=True. Inverse: days leftover after size OLS 0.666 rank 0.667 dies=False.

OLS folds: 0.533 0.532 0.420 0.593 0.576. Rank folds: 0.496 0.523 0.610 0.448 0.527.

## 4 — Same-n leftover

same-n n=5528. log leftover 0.521; raw a_in3 leftover 0.521; a_op_in leftover 0.584. raw leftover after log1p 0.611 dies=True; a_op_in leftover after log1p 0.627 dies=True.

| feature | n | n_pos | CV | leftover rank | leftover OLS | dies |
| --- | --- | --- | --- | --- | --- | --- |
| log1p(a_in3) | 5528 | 391 | 0.617 | 0.521 | 0.531 | True |
| a_in3 | 5528 | 391 | 0.617 | 0.521 | 0.694 | True |
| a_op_in | 5528 | 391 | 0.679 | 0.584 | 0.690 | True |
| log1p(a_op_in) | 5528 | 391 | 0.679 | 0.584 | 0.568 | False |


## 5 — SIZE terciles (T1 inverse-size)

T1 leftover after days rank 0.534 dies=True. Inverse-size recover lives inside T1 only if leftover lives there.

| tercile | n | n_pos | size CV | days CV | leftover rank | dies | Y3 rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 1298 | 215 | 0.488 | 0.594 | 0.534 | True | 16.7% |
| T2 | 2043 | 115 | 0.557 | 0.613 | 0.634 | False | 5.6% |
| T3 | 2187 | 61 | 0.612 | 0.624 | 0.662 | False | 2.9% |


## 6 — ICC / demean

ICC=0.969 TRAIT k=1214. Demean CV 0.627 leftover-days 0.640 dies=False. Company-mean CV 0.754 leftover-days 0.676 dies=False.

## 7 — Q6 lag1 leftover after days_lag1

Y3 log_in3_lag1 0.608 leftover after days_lag1 rank 0.527 dies=True. lag3 0.629 leftover after days_lag3 rank 0.555 dies=False. a_growth_12 already CLOSE as Q6 — not rewritten.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | log_in3 | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | log_in3_lag1 | 5,078 | 355 | 0.608 | 0.614 | 0.080 | -1 | 0.508 0.643 0.691 0.539 0.659 |
| y3_recover_cash_6m | log_in3_lag3 | 4,212 | 313 | 0.629 | 0.638 | 0.077 | -1 | 0.507 0.665 0.687 0.599 0.686 |


## 9 — Holdout coverage only

Holdout 72 co / 1073 CM nn=929 cov=86.6% p50=339930 (no fit, no AUROC).

## 10 — Dark 470 vs ERP

Dark 470 co size Y3 0.644; ERP 0.623.

| book | n_co | n_cm | Y3 n | Y3 pos | size CV | leftover rank | dies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dark | 470 | 7603 | 1985 | 134 | 0.644 | 0.457 | True |
| ERP | 744 | 13554 | 3543 | 257 | 0.623 | 0.531 | True |


## Extras

### a_in6 / a_in12 twins

a_in6 leftover after log_in3 0.562 dies=False; a_in12 0.437 dies=True. Do not add a_in6/a_in12 besides a_in3.

| feature | CV | leftover after log_in3 | dies vs size | leftover after days | dies vs days |
| --- | --- | --- | --- | --- | --- |
| log_in6 | 0.601 | 0.562 | False | 0.452 | True |
| log_in12 | 0.557 | 0.437 | True | 0.515 | True |


### leftover vs salary

size leftover after salary rank 0.563 dies=False. size leftover after days+salary 0.474 dies=True. salary leftover after size 0.634 dies=False (salary_month_qa leftover after days was 0.603 — not overwritten).

### Fold-wise leftover; 12-name Y2 drop

Y2 size 0.552 days 0.571. Drop 12 chronic: size 0.545 days 0.549. Y2 leftover after days 0.466. Y3 leftover wo12 0.524 dies=True.

| fold | OLS leftover | rank leftover | n_va | n_pos |
| --- | --- | --- | --- | --- |
| 0 | 0.533 | 0.496 | 1288 | 53 |
| 1 | 0.532 | 0.523 | 685 | 92 |
| 2 | 0.420 | 0.610 | 1047 | 64 |
| 3 | 0.593 | 0.448 | 1342 | 78 |
| 4 | 0.576 | 0.527 | 1166 | 104 |


| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | log_in3 | 14,968 | 1,044 | 0.552 | 0.540 | 0.046 | 1 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | days | 17,356 | 1,271 | 0.571 | 0.577 | 0.046 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | log_in3 wo12 | 14,776 | 881 | 0.545 | 0.537 | 0.054 | 1 | 0.487 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | days wo12 | 17,140 | 1,094 | 0.549 | 0.551 | 0.041 | 1 | 0.513 0.539 0.612 0.565 0.517 |


### Bootstrap leftover after days

Bootstrap leftover-after-days rank p50=0.542 p10=0.439 p90=0.583 share<0.55=58.0% n=50.

### Permute within days quintile

Permuted-within-days leftover rank p50=0.518 p90=0.534 n=30.

### OLS fake-days leak

OLS leftover 0.531 ρ(resid,days)=-0.163 fake=False not a days leak. Rank leftover 0.521 dies=True.

### Demean leftover vs level leftover

Demean Y3 0.627 leftover-days 0.640 dies=False. Mean Y3 0.754 leftover-days 0.676 dies=False. Demean leftover after days+mean 0.631 dies=False. Level leftover 0.521 dies — the contemporaneous size bar is unused after days. Demean leftover is a STATE path, not a reason to KEEP a_in3 as engine X.

### Leftover by so-far

Leftover after days by so-far bucket.

| so-far | n | n_pos | raw | rank leftover | dies |
| --- | --- | --- | --- | --- | --- |
| <6 | 1316 | 78 | 0.635 | 0.576 | False |
| 6-11 | 2287 | 163 | 0.582 | 0.422 | True |
| 12-17 | 1712 | 134 | 0.574 | 0.513 | True |
| 18-23 | 213 | 16 | — | — | False |
| 24+ | 0 | 0 | — | — | False |


### a_op_in SIZE clone

a_op_in Y3 0.676 leftover after days 0.584 dies=True. leftover after log_in3 0.627 dies=True. ρ vs log_in3 0.876 acf1=0.004. 0.676 single is monthly extremes (acf≈0), not a swap for log1p(a_in3). Do not put a_op_in on the 44.

### Without weakest / strongest fold

Size without weakest fold 3 = 0.635; without strongest 2 = 0.600. Rank leftover without fold 0 = 0.527 (still dies).

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `a_in3` on the 15-col card.

## Files written

- `analysis/evaluate/in3_qa.py`
- `analysis/outputs/in3_qa.md`
- `analysis/outputs/in3_leftover.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_in3.md` (end, if WRITE_WAVE)

Elapsed 39s. Failed: none.

