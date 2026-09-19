# Unused leftover of `c_salary_month` after days

Generated `2026-09-19T05:32:30+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_salary`. Do not overwrite `salary_qa.py`. Do not grow TURNOVER. Do not change the 15-col Y3 card.

`c_salary_month` = 1 if any category = salary this month (fillna 0). Still on the 15-col Y3 card (quoted 0.671). `c_missed_salary` already CLOSE as Y3 X (0.513). `f_ds_r` and `a_n_tx` were DROPPED from the card. This cut asks leftover after days.

## Headline

`c_salary_month` on the 15-col card: **KEEP** (leftover after days rank 0.603 lives, beats size by 0.055, not SIZE, not a twin). Y3 leftover after days OLS 0.603 rank 0.603 (lives, fake=False). Inverse days after salary rank 0.647. After SS 0.626; after missed 0.671. Single 0.671 vs days 0.711 vs size 0.617 beat_size=True. Calendar monthly. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Presence salary is a quiet/busy **flag**, already a card stem. KEEP after days. |
| 2 | Who is improving? | Q6 lag leftover after days_lag1 is the path test. |
| 3 | Who is turning? | Not a new Y. Do not invent `y_salary`. |
| 4 | Dip vs fall? | Not a TURNOVER column. Do not grow 0.720. |
| 5 | Why did it change? | Leftover after days rank 0.603. After SS 0.626. Twin: none. |
| 6 | Months earlier? | lag1 leftover after days_lag1 0.606; lag3 0.587. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `c_salary_month` on the 15-col card | **KEEP** | leftover after days rank 0.603 lives, beats size by 0.055, not SIZE, not a twin |
| calendar dummy | **monthly** | salary Q-month 40.5% vs other 39.9% gap=0.6% → monthly. tax Q 68.9% vs other 43.1% gap=25.9%. is_q Y3 0.495 is_aug 0.506. salary leftover after is_q rank 0.672 dies=False. |
| `c_missed_salary` as Y3 X | **CLOSE** | already 0.513; not rewritten |

## 1 — Coverage; modal; acf1; size ρ

Train 1,214 co / 21,157 CM. salary cov 100.0% share=1 40.1% modal=0 59.9%. ACF1=0.092. ρ vs log_in3 0.333 SIZE=False.

| col | n_nn | cov | share=1 | modal=0 | acf1 | ρ size | SIZE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c_salary_month | 21,157 | 100.0% | 40.1% | 59.9% | 0.092 | 0.333 | False |


Holdout coverage only: 72 co / 1073 CM nn=1073 cov=100.0% share=1=37.8%.

## 2 — Spearman twins

ρ vs days 0.419 vs a_n_tx 0.413 vs SS 0.619 vs missed -0.143 vs tax 0.317. Twins (≥0.80): none. salary∩SS Jaccard 0.639 both=7,075.

| vs | rho | twin |
| --- | --- | --- |
| c_n_days_with_tx | 0.419 | no |
| a_n_tx | 0.413 | no |
| c_ss_month | 0.619 | no |
| c_missed_salary | -0.143 | no |
| c_tax_month | 0.317 | no |
| log_in3 | 0.333 | no |


## 3 — Single-feature group-fold AUROC

Y3 salary 0.671 vs days 0.711 vs size 0.617. Replica days 0.711 CONFIRM / size 0.617 CONFIRM / salary 0.671 CONFIRM. missed 0.513 CONFIRM 0.513. Beat size ≥0.02: True (Δ=0.055).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | c_salary_month | 5,648 | 402 | 0.671 | 0.672 | 0.050 | -1 | 0.611 0.659 0.745 0.689 0.652 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log_in3 | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_ss_month | 5,648 | 402 | 0.693 | 0.690 | 0.031 | -1 | 0.673 0.667 0.745 0.687 0.694 |
| y3_recover_cash_6m | c_missed_salary | 5,648 | 402 | 0.513 | 0.512 | 0.010 | 1 | 0.531 0.510 0.507 0.507 0.508 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | c_tax_month | 5,648 | 402 | 0.611 | 0.617 | 0.031 | -1 | 0.556 0.624 0.630 0.613 0.630 |
| y2_neg_2of3 | c_salary_month | 17,356 | 1,271 | 0.522 | 0.513 | 0.055 | 1 | 0.511 0.572 0.535 0.432 0.559 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.577 | 0.046 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | log_in3 | 14,968 | 1,044 | 0.552 | 0.540 | 0.046 | 1 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | c_ss_month | 17,356 | 1,271 | 0.539 | 0.531 | 0.043 | -1 | 0.509 0.562 0.516 0.604 0.505 |
| y2_neg_2of3 | c_missed_salary | 17,356 | 1,271 | 0.494 | 0.503 | 0.009 | 1 | 0.508 0.485 0.495 0.486 0.496 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 0.601 | 0.044 | 1 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | c_tax_month | 17,356 | 1,271 | 0.535 | 0.541 | 0.026 | 1 | 0.559 0.558 0.540 0.520 0.498 |


## 4 — Honest leftover after days

salary leftover after days OLS 0.603 rank 0.603 fake=False ρ(resid,days)=-0.117 R²=0.174 honest_dies=False. Inverse: days leftover after salary OLS 0.649 rank 0.647 dies=False.

OLS folds: 0.550 0.573 0.708 0.618 0.567. Rank folds: 0.550 0.573 0.708 0.618 0.567.

## 5 — Leftover after SS / missed

after SS OLS 0.626 rank 0.626 dies=False R²=0.384. after missed OLS 0.671 rank 0.671 dies=False. after days+SS rank 0.591 dies=False.

## 6 — SIZE terciles; ICC / demean

ICC=0.978 TRAIT k=1214. Demean CV 0.542 company-mean 0.730. Demean leftover after days rank 0.530 dies=True.

| tercile | n | n_pos | CV | leftover rank | dies |
| --- | --- | --- | --- | --- | --- |
| T1 | 1,331 | 222 | 0.554 | 0.399 | True |
| T2 | 2,084 | 116 | 0.650 | 0.620 | False |
| T3 | 2,233 | 64 | 0.667 | 0.653 | False |


## 7 — Q6 lag1 / lag3 leftover after days_lag

Y3 lag1 0.663 leftover after days_lag1 rank 0.606 dies=False. lag3 0.639 leftover after days_lag3 rank 0.587 dies=False. contemp leftover after days_lag1 rank 0.617.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | c_salary_month | 5,648 | 402 | 0.671 | 0.672 | 0.050 | -1 | 0.611 0.659 0.745 0.689 0.652 |
| y3_recover_cash_6m | c_salary_month_lag1 | 5,648 | 402 | 0.663 | 0.660 | 0.050 | -1 | 0.628 0.648 0.739 0.685 0.614 |
| y3_recover_cash_6m | c_salary_month_lag3 | 5,078 | 355 | 0.639 | 0.639 | 0.048 | -1 | 0.590 0.626 0.709 0.665 0.606 |


## 8 — Calendar dummy

salary Q-month 40.5% vs other 39.9% gap=0.6% → monthly. tax Q 68.9% vs other 43.1% gap=25.9%. is_q Y3 0.495 is_aug 0.506. salary leftover after is_q rank 0.672 dies=False.

| month | n | salary | tax | Y3 rate |
| --- | --- | --- | --- | --- |
| 1 | 1768 | 39.5% | 67.9% | 6.3% |
| 2 | 1881 | 40.0% | 39.5% | 6.9% |
| 3 | 1925 | 41.2% | 43.1% | 6.8% |
| 4 | 1945 | 41.4% | 68.6% | 6.1% |
| 5 | 1966 | 39.2% | 43.7% | 6.5% |
| 6 | 1976 | 39.6% | 42.5% | 8.0% |
| 7 | 2010 | 40.3% | 67.9% | 9.4% |
| 8 | 2047 | 39.4% | 38.6% | 8.3% |
| 9 | 1299 | 40.1% | 45.0% | 6.5% |
| 10 | 1381 | 40.7% | 72.4% | 7.6% |
| 11 | 1428 | 40.1% | 46.1% | 7.2% |
| 12 | 1531 | 39.6% | 48.9% | 6.6% |


## 10 — Dark 470 vs invoiced

Dark 470 co salary share 36.0% Y3 0.631; ERP share 42.4% Y3 0.705.

| book | n_co | n_cm | salary share | Y3 n | Y3 pos | CV | leftover rank | dies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dark | 470 | 7603 | 36.0% | 2030 | 138 | 0.631 | 0.547 | True |
| ERP | 744 | 13554 | 42.4% | 3618 | 264 | 0.705 | 0.634 | False |


## Extras

### salary ∩ SS Jaccard

salary∩SS Jaccard 0.639.

| vs | jaccard | n_sal | n_other | n_both |
| --- | --- | --- | --- | --- |
| c_ss_month | 0.639 | 8484 | 9665 | 7075 |
| c_tax_month | 0.451 | 8484 | 10953 | 6037 |
| c_missed_salary | 0.000 | 8484 | 623 | 0 |


### leftover after tax

salary leftover after tax rank 0.641 dies=False. tax leftover after salary rank 0.538 dies=True.

### Fold-wise leftover; 12-name Y2 drop

Y2 salary 0.522 days 0.571. Drop 12 chronic: salary 0.498 days 0.549. Y2 leftover after days 0.459 wo12 0.448. Y3 leftover wo12 0.607 dies=False.

| fold | OLS leftover | rank leftover | n_va | n_pos |
| --- | --- | --- | --- | --- |
| 0 | 0.550 | 0.550 | 1310 | 54 |
| 1 | 0.573 | 0.573 | 696 | 93 |
| 2 | 0.708 | 0.708 | 1072 | 66 |
| 3 | 0.618 | 0.618 | 1373 | 82 |
| 4 | 0.567 | 0.567 | 1197 | 107 |


| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | c_salary_month | 17,356 | 1,271 | 0.522 | 0.513 | 0.055 | 1 | 0.511 0.572 0.535 0.432 0.559 |
| y2_neg_2of3 | days | 17,356 | 1,271 | 0.571 | 0.577 | 0.046 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | salary wo12 | 17,140 | 1,094 | 0.498 | 0.510 | 0.060 | 1 | 0.507 0.572 0.535 0.432 0.441 |
| y2_neg_2of3 | days wo12 | 17,140 | 1,094 | 0.549 | 0.551 | 0.041 | 1 | 0.513 0.539 0.612 0.565 0.517 |


### Holdout coverage

Holdout 72 co / 1073 CM nn=1073 cov=100.0% share=1=37.8% (no fit).

### Bootstrap leftover after days

Bootstrap leftover-after-days rank p50=0.604 p10=0.581 p90=0.628 share<0.55=1.7% n=60.

### OLS fake-days leak

OLS leftover 0.603 ρ(resid,days)=-0.117 fake=False not a days leak. Rank leftover 0.603 dies=False.

### leftover after days+size / a_n_tx

after days+size rank 0.603 dies=False. after a_n_tx rank 0.604 dies=False.

### SS leftover after salary

SS leftover after salary rank 0.668 dies=False R²=0.384. SS leftover after days rank 0.635 dies=False. SS is in-flight leftover — do not touch ss_qa.

### salary × days tercile

Y3 rate by salary × days tercile — leftover should show inside D1/D2/D3.

| days tercile | salary | n | n_pos | Y3 rate | med days |
| --- | --- | --- | --- | --- | --- |
| D1 | no salary | 1460 | 228 | 15.6% | 5.000 |
| D1 | salary | 588 | 43 | 7.3% | 10.000 |
| D2 | no salary | 719 | 63 | 8.8% | 19.000 |
| D2 | salary | 1161 | 24 | 2.1% | 19.000 |
| D3 | no salary | 469 | 26 | 5.5% | 25.000 |
| D3 | salary | 1251 | 18 | 1.4% | 26.000 |


### Permute within days quintile

Permuted-within-days leftover rank p50=0.516 p90=0.528 n=40. Observed leftover 0.603 should sit above this null if leftover is real.

### Leftover by company trail length

Leftover after days by company trail length.

| book | n | n_pos | raw | rank leftover | dies |
| --- | --- | --- | --- | --- | --- |
| short_<12 | 146 | 13 | — | — | False |
| mid_12_17 | 364 | 34 | — | — | False |
| long_>=18 | 5138 | 355 | 0.666 | 0.596 | False |


### Dark leftover folds

Dark leftover folds OLS 0.485 0.556 0.474 0.561 0.660 rank 0.485 0.556 0.474 0.561 0.660 rank=0.547 dies=True. ERP leftover rank 0.634 dies=False. KEEP is ERP-weighted; dark leftover dies.

### Company-mean leftover after days

Company-mean salary Y3 0.730 leftover after days rank 0.662 dies=False. Demean leftover died at 0.530 — leftover is mostly TRAIT (who pays salary), but the days-tercile 2×2 still shows a same-month STATE gap.

### Leftover by so-far

Leftover after days by so-far bucket.

| so-far | n | n_pos | raw | rank leftover | dies |
| --- | --- | --- | --- | --- | --- |
| <6 | 1436 | 89 | 0.688 | 0.667 | False |
| 6-11 | 2287 | 163 | 0.665 | 0.583 | False |
| 12-17 | 1712 | 134 | 0.677 | 0.636 | False |
| 18-23 | 213 | 16 | — | — | False |
| 24+ | 0 | 0 | — | — | False |


### Without strongest fold

Y3 salary without strongest fold 2 = 0.653. Rank leftover without strongest fold 2 = 0.577 (still lives).

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put new cols on the 15-col card.

## Files written

- `analysis/evaluate/salary_month_qa.py`
- `analysis/outputs/salary_month_qa.md`
- `analysis/outputs/salary_month_leftover.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_salary_month.md` (end, if WRITE_WAVE)

Elapsed 45s. Failed: none.

