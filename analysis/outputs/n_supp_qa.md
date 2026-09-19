# Unused leftover of `d_n_supp` after days as Y3 X

Generated `2026-09-19T05:56:47+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_n_supp`. Do not put `d_n_supp` on the 15-col card. Do not overwrite `supp_hhi_qa.*` or `n_cust_qa.*`. Do not grow TURNOVER. Y5 leftover after size is report-only.

`d_n_supp` = distinct non-null counterparty_id on AP invoices (Family D, trailing 6m). Incomplete 6m books are NaN. Dark 470 stay NaN not 0. `d_n_cust` CLOSED leftover 0.545 / DROP from 44. `d_supp_hhi` already DROP (twin of top1 ρ 0.987).

## Headline

`d_n_supp` as Y3 X: **DROP from the 44 as Y3 X** (leftover after days rank 0.587 lives, beat-size PASS, not a twin, but SIZE (ρ vs log1p(a_in3)=0.546 ≥0.50). KEEP-as-X fails the SIZE gate. Off the 15-col card. Do not invent y_n_supp.). Y3 leftover after days OLS 0.530 rank 0.587 (lives, fake=False). Inverse days after n_supp rank 0.666. Single 0.699 vs size 0.617 vs days 0.711 vs top1 0.641 vs HHI 0.653 vs n_cust 0.653. after top1 0.639 after HHI 0.626 after n_cust 0.649 after days+top1 0.543. 15-col card: no — do not put d_n_supp on the 15-col card. PARK as Y — do not invent y_n_supp. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — do not invent y_n_supp. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Q6 lag1 leftover after days_lag1 0.598. |
| 3 | Who is turning? | **DROP from the 44 as Y3 X** leftover after days 0.587 vs days 0.711. |
| 4 | Dip vs fall? | Y5 leftover after size 0.561 — report-only. |
| 5 | Why did it change? | Twin screen: none. ρ vs n_cust 0.632. |
| 6 | Months earlier? | lag1 leftover 0.598; days_lag1 0.684 (quote 0.684). |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `d_n_supp` as Y3 X / the 15-col card | **DROP from the 44 as Y3 X** | leftover after days rank 0.587 lives, beat-size PASS, not a twin, but SIZE (ρ vs log1p(a_in3)=0.546 ≥0.50). KEEP-as-X fails the SIZE gate. Off the 15-col card. Do not invent y_n_supp. |
| `d_n_supp` as engine X on the 44 | **DROP** | leftover lives=True twin=False SIZE=True beat-size=True |
| `y_n_supp` | **PARK** | do not invent a supplier-count Y |
| twin of HHI / top1 / n_cust | -0.726 / -0.661 / 0.632 | leftover after top1 0.639 after HHI 0.626 after n_cust 0.649 |
| same object as `d_n_cust` | **NO** | ρ=0.632 (peek 0.632) |
| Q6 lag1 after days_lag1 | **KEEP** | leftover 0.598; days_lag1 0.684 |
| Y5 leftover after size | report-only | 0.561 dies=False; never E |

## 1 — Coverage; twin / SIZE screen

Train 1,214 co / 21,157 CM. d_n_supp cov 53.4% n0=698 (HHI defined on n0=0). acf1=0.906. Dark 470 (want 470) nn=0 zero=0 CONFIRM NaN. ρ vs days 0.607 vs a_n_tx 0.639 vs top1 -0.661 vs HHI -0.726 vs n_cust 0.632 (peek 0.632 CONFIRM) vs size 0.546. SIZE=True twins=none twin_gate=False.

| col | n_nn | cov | n0 | acf1 |
| --- | --- | --- | --- | --- |
| d_n_supp | 11,293 | 53.4% | 698 | 0.906 |
| d_supp_hhi | 10,595 | 50.1% | — | 0.745 |
| d_supp_top1 | 10,595 | 50.1% | — | 0.711 |
| d_n_cust | 11,293 | 53.4% | — | 0.867 |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | 0.607 | no |
| a_n_tx | 0.639 | no |
| d_supp_top1 | -0.661 | no |
| d_supp_hhi | -0.726 | no |
| d_n_cust | 0.632 | no |
| log1p(a_in3) | 0.546 | SIZE |
| d_cust_hhi | -0.319 | no |
| d_cust_top1 | -0.304 | no |
| d_tx_cp_share | 0.245 | no |


## 2 — Single-feature group-fold Y3

Y3 d_n_supp 0.699 n=3,003 pos=221 (peek 0.699 / 3,003 / 221 CONFIRM). vs size 0.617 vs days 0.711 vs top1 0.641 vs HHI 0.653 vs n_cust 0.653 (peek 0.653 CONFIRM). Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=0.082 PASS.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_n_supp | 3,003 | 221 | 0.699 | 0.685 | 0.116 | -1 | 0.820 0.722 0.514 0.764 0.674 |
| y3_recover_cash_6m | log1p(d_n_supp) | 3,003 | 221 | 0.699 | 0.685 | 0.116 | -1 | 0.820 0.722 0.514 0.764 0.674 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | d_supp_top1 | 2,877 | 203 | 0.641 | 0.605 | 0.102 | 1 | 0.787 0.702 0.561 0.613 0.542 |
| y3_recover_cash_6m | d_supp_hhi | 2,877 | 203 | 0.653 | 0.623 | 0.091 | 1 | 0.789 0.689 0.573 0.643 0.570 |
| y3_recover_cash_6m | d_n_cust | 3,003 | 221 | 0.653 | 0.650 | 0.077 | -1 | 0.742 0.662 0.530 0.665 0.666 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |


## 3 — Honest leftover after days

d_n_supp leftover after days OLS 0.530 rank 0.587 (n_cust peek 0.587 CONFIRM) fake=False almost=False ρ(resid,days)=-0.275 R²=0.103 honest_dies=False n=3,003 pos=221. Inverse: days leftover after n_supp OLS 0.738 rank 0.666 dies=False. log1p leftover rank 0.587.

OLS folds: 0.431 0.588 0.566 0.533 0.533. Rank folds: 0.733 0.544 0.462 0.637 0.561.

## 4 — Twin / SIZE screen (in cut 1)

SIZE=True twin_gate=False twins=none.

## 5 — Leftover after top1 / HHI / n_cust / days+top1

n_supp leftover after top1 rank 0.639 dies=False; after HHI 0.626 dies=False; after n_cust 0.649 dies=False; after days+top1 0.543 dies=True; after days+n_cust 0.581 dies=False. Inverse n_cust after n_supp 0.535 (n_cust leftover after days was 0.545 — not overwritten).

| bar | OLS | rank | ρ(resid,bar) | R2 | dies | n | n_pos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| after top1 | 0.559 | 0.639 | 0.285 | 0.130 | False | 2877 | 203 |
| after HHI | 0.567 | 0.626 | 0.167 | 0.121 | False | 2877 | 203 |
| after n_cust | 0.693 | 0.649 | 0.499 | 0.009 | False | 3003 | 221 |
| after days+top1 | 0.551 | 0.543 | -0.204 | 0.167 | True | 2877 | 203 |
| after days+n_cust | 0.529 | 0.581 | -0.275 | 0.104 | False | 3003 | 221 |
| top1 after n_supp | 0.629 | 0.396 | -0.517 | 0.130 | True | 2877 | 203 |
| HHI after n_supp | 0.637 | 0.401 | -0.556 | 0.121 | True | 2877 | 203 |
| n_cust after n_supp | 0.556 | 0.535 | 0.221 | 0.009 | True | 3003 | 221 |


## 6 — Dark 470 stay NaN; ERP-only leftover

Dark 470 (want 470) n_supp nn=0 zero=0 CONFIRM NaN not 0. ERP Y3 0.699 leftover after days rank 0.587 dies=False n=3,003 pos=221.

| book | n_co | n_cm | n_supp nn | Y3 n | Y3 pos | CV | leftover rank | dies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dark | 470 | 7603 | 0 | 0 | 0 | LOW_POWER | — | — |
| ERP | 744 | 13554 | 11293 | 3003 | 221 | 0.699 | 0.587 | False |


## 7 — Q6 lag1 leftover after days_lag1

Y3 n_supp_lag1 0.692 leftover after days_lag1 rank 0.598 dies=False. lag3 leftover after days_lag3 0.605 dies=False. Days lag1 0.684 (quote 0.684 CONFIRM). Short Y3 lag1 leftover 0.572 n=1,544 pos=105.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_n_supp | 3,003 | 221 | 0.699 | 0.685 | 0.116 | -1 | 0.820 0.722 0.514 0.764 0.674 |
| y3_recover_cash_6m | d_n_supp_lag1 | 2,804 | 209 | 0.692 | 0.679 | 0.095 | -1 | 0.809 0.709 0.547 0.723 0.671 |
| y3_recover_cash_6m | d_n_supp_lag3 | 2,274 | 171 | 0.682 | 0.672 | 0.081 | -1 | 0.797 0.690 0.571 0.663 0.688 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | 0.694 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |


## 8 — Y5 leftover after size (report-only); protective tail

Y5 n_supp 0.584 vs size 0.556 vs top1 0.530 vs HHI 0.539. Leftover after size rank 0.561 dies=False (report-only; Y5 never E). HHI>0.975 Y5 rate 2.7% (quote 2.7% CONFIRM) vs rest 8.6% (quote 8.6% CONFIRM). n_supp==1 rate 0.0% vs ≥2 8.6%. Share of tail with n_supp==1 50.7%. not just a tail rewrite.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y5_ap_od30_ownp80 | d_n_supp | 4,905 | 418 | 0.584 | 0.561 | 0.064 | 1 | 0.676 0.560 0.566 0.506 0.612 |
| y5_ap_od30_ownp80 | log1p(a_in3) | 4,905 | 418 | 0.556 | 0.545 | 0.084 | 1 | 0.674 0.539 0.466 0.495 0.606 |
| y5_ap_od30_ownp80 | d_supp_top1 | 4,902 | 418 | 0.530 | 0.518 | 0.043 | -1 | 0.526 0.565 0.523 0.464 0.571 |
| y5_ap_od30_ownp80 | d_supp_hhi | 4,902 | 418 | 0.539 | 0.526 | 0.044 | -1 | 0.536 0.575 0.533 0.470 0.579 |


| bar | rank | OLS | dies | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| n_supp after size | 0.561 | 0.557 | False | 4905 | 418 |
| n_supp after top1 | 0.577 | 0.560 | False | 4902 | 418 |
| n_supp after HHI | 0.571 | 0.559 | False | 4902 | 418 |


## 9 — vs `d_n_cust`

ρ(n_supp, n_cust)=0.632 (peek 0.632 CONFIRM) different object. Y3 n_cust 0.653 leftover after days 0.545 (n_cust_qa 0.545 — not overwritten). n_supp leftover after n_cust 0.649 dies=False. n_cust leftover after n_supp 0.535 dies=True.

## 10 — Holdout coverage only

Holdout 72 co / 1073 CM nn=542 cov=50.5% n0=15 p50=19.0 dark nn=0 (no fit, no AUROC).

## Extras

### ICC / demean

ICC=0.965 TRAIT k=744. Demean CV 0.473 leftover-days 0.518 dies=True. Company-mean CV 0.725 leftover-days 0.622 dies=False.

### n==0 / n==1 / ≥2

Drop n==0: Y3 0.690 leftover after days 0.569 dies=False n=2,877.

| bin | n | n_pos | Y3 rate |
| --- | --- | --- | --- |
| n==0 | 126 | 18 | 14.3% |
| n==1 | 83 | 12 | 14.5% |
| n 2-5 | 420 | 65 | 15.5% |
| n>=6 | 2374 | 126 | 5.3% |


### Fold-wise leftover; 12-name Y2 drop

Y2 n_supp 0.567 days 0.571. Drop 12 chronic: 0.567. Y2 leftover after days 0.423. Y3 leftover wo12 0.587 dies=False.

| fold | OLS leftover | rank leftover | n_va | n_pos |
| --- | --- | --- | --- | --- |
| 0 | 0.431 | 0.733 | 455 | 26 |
| 1 | 0.588 | 0.544 | 244 | 38 |
| 2 | 0.566 | 0.462 | 681 | 39 |
| 3 | 0.533 | 0.637 | 813 | 51 |
| 4 | 0.533 | 0.561 | 810 | 67 |


### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.564 p50=0.607 p95=0.654 share<0.55=4.0% n=50.

### Leftover by so-far

Leftover after days by so-far bucket.

| so-far | n | n_pos | raw | rank leftover | dies |
| --- | --- | --- | --- | --- | --- |
| <6 | 371 | 27 | — | — | False |
| 6-11 | 1366 | 90 | 0.714 | 0.570 | False |
| 12-17 | 1130 | 93 | 0.710 | 0.624 | False |
| 18-23 | 136 | 11 | — | — | False |
| 24+ | 0 | 0 | — | — | False |


### log1p vs raw

same-n raw 0.699 leftover 0.587; log1p 0.699 leftover 0.587.

### Permute within days quintile

Permuted-within-days leftover rank p50=0.542 p90=0.570 n=30.

### leftover after a_n_tx

n_supp leftover after a_n_tx rank 0.582 dies=False. after days+a_n_tx 0.581 dies=False.

### leftover after size

n_supp leftover after size rank 0.657 dies=False. after days+size 0.591 dies=False. size leftover after n_supp 0.523 dies=True (in3 leftover after days was 0.521 — not overwritten).

### Without weakest / strongest fold

n_supp without weakest fold 2 = 0.745; without strongest 0 = 0.669. Rank leftover without fold 0 = 0.551 (still lives).

### leftover after days on top1-defined only

top1-defined same-n n=2,877 pos=203. raw 0.690 days 0.744. leftover after days 0.569 dies=False. after top1 0.639 dies=False. after days+top1 0.543 dies=True.

### SIZE tercile leftover

SIZE tercile leftover of n_supp after days.

| tercile | n | n_pos | CV | leftover rank | dies | Y3 rate |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | 692 | 127 | 0.575 | 0.516 | True | 16.7% |
| T2 | 1138 | 59 | 0.643 | 0.561 | False | 5.6% |
| T3 | 1173 | 35 | LOW_POWER | — | False | 2.9% |


### Y3 rate by n_supp quintile

Y3 n_supp Q1→Q5 ['15.1%', '8.9%', '6.3%', '3.4%', '2.5%']; days Q1→Q5 ['19.4%', '7.5%', '3.3%', '3.8%', '2.0%'].

| q | n_supp rate | n | n_pos |
| --- | --- | --- | --- |
| 1 | 15.1% | 629 | 95 |
| 2 | 8.9% | 609 | 54 |
| 3 | 6.3% | 586 | 37 |
| 4 | 3.4% | 583 | 20 |
| 5 | 2.5% | 596 | 15 |


### SIZE leftover probe

ρ(n_supp, log_in3)=0.546 SIZE_gate=True. n_supp leftover after size rank 0.657 dies=False ρ(resid,size)=-0.061 folds=0.755 0.693 0.481 0.738 0.618. after days+size 0.591 dies=False. size leftover after n_supp 0.523 dies=True. Leftover after size lives — not a size clone; SIZE gate is |ρ|≥0.50 only.

### Y5 n==1 vs protective tail

Y5 n==1 n=37 pos=0 rate 0.0%. tail∩n==1 37. n==1 dummy Y5 0.504 leftover after size 0.550 after top1 0.477. Report-only.

### leftover after days vs days+top1

after days 0.587 lives; after top1 0.639 lives; after days+top1 0.543 dies=True folds=0.563 0.375 0.473 0.689 0.615. after days+size 0.591 dies=False. Days+top1 eat the leftover — count leftover after days is the activity+concentration stack, not a new engine X. SIZE leftover after days still lives so SIZE is not the eater.

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| days_lag1 | 0.684 |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `d_n_supp` on the 15-col card.

## Files written

- `analysis/evaluate/n_supp_qa.py`
- `analysis/outputs/n_supp_qa.md`
- `analysis/outputs/n_supp_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_n_supp.md` (end, if WRITE_WAVE)

Elapsed 41s. Failed: none.

