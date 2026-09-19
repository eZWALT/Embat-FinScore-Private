# Unused leftover of `d_n_cust` after days as Y3 X

Generated `2026-09-19T05:49:18+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_n_cust`. Do not put `d_n_cust` on the 15-col card. Do not overwrite `cust_hhi_qa.*`. Do not grow TURNOVER.

`d_n_cust` = distinct non-null counterparty_id on AR invoices (Family D, trailing 6m). Incomplete 6m books are NaN. Dark 470 stay NaN not 0. `d_cust_hhi` was DROPPED as engine X (twin of `d_cust_top1` ρ 0.994).

## Headline

`d_n_cust` as Y3 X: **CLOSE unused leftover** (unused leftover after days: honest rank 0.545 dies (OLS 0.648 fake=False almost=True). Also TWIN of ['d_cust_top1', 'd_cust_hhi']. DROP from the 44 as Y3 X. Do not invent y_n_cust. Off the 15-col card.). Y3 leftover after days OLS 0.648 rank 0.545 (dies, fake=False). Inverse days after n_cust rank 0.699. Single 0.653 vs size 0.617 vs days 0.711 vs top1 0.590 vs HHI 0.595. after top1 0.456 after HHI 0.459 after days+top1 0.564. 15-col card: no — do not put d_n_cust on the 15-col card. PARK as Y — do not invent y_n_cust. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — do not invent y_n_cust. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Q6 lag1 leftover after days_lag1 0.539. |
| 3 | Who is turning? | **CLOSE unused leftover** leftover after days 0.545 vs days 0.711. |
| 4 | Dip vs fall? | Y4 leftover after top1_lag3 0.530 — monopoly tail rewrite? |
| 5 | Why did it change? | Twin screen: ['d_cust_top1', 'd_cust_hhi']. ρ vs HHI -0.837. |
| 6 | Months earlier? | lag1 leftover 0.539; days_lag1 0.684 (quote 0.684). |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `d_n_cust` as Y3 X / the 15-col card | **CLOSE unused leftover** | unused leftover after days: honest rank 0.545 dies (OLS 0.648 fake=False almost=True). Also TWIN of ['d_cust_top1', 'd_cust_hhi']. DROP from the 44 as Y3 X. Do not invent y_n_cust. Off the 15-col card. |
| `d_n_cust` as engine X on the 44 | **DROP** | leftover lives=False twin=True SIZE=False beat-size=True |
| `y_n_cust` | **PARK** | do not invent a customer-count Y |
| twin of HHI / top1 | -0.837 / -0.806 | leftover after top1 0.456 after HHI 0.459 |
| same object as `d_n_supp` | **NO** | ρ=0.632 |
| Q6 lag1 after days_lag1 | **CLOSE** | leftover 0.539; days_lag1 0.684 |

## 1 — Coverage; twin / SIZE screen

Train 1,214 co / 21,157 CM. d_n_cust cov 53.4% n0=2,365 (HHI defined on n0=0). acf1=0.867. Dark 470 (want 470) nn=0 zero=0 CONFIRM NaN. ρ vs days 0.507 vs a_n_tx 0.526 vs top1 -0.806 vs HHI -0.837 (peek -0.837 CONFIRM) vs n_supp 0.632 vs size 0.381. SIZE=False twins=['d_cust_top1', 'd_cust_hhi'] twin_gate=True.

| col | n_nn | cov | n0 | acf1 |
| --- | --- | --- | --- | --- |
| d_n_cust | 11,293 | 53.4% | 2,365 | 0.867 |
| d_cust_hhi | 8,928 | 42.2% | — | 0.768 |
| d_cust_top1 | 8,928 | 42.2% | — | 0.737 |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | 0.507 | no |
| a_n_tx | 0.526 | no |
| d_cust_top1 | -0.806 | TWIN |
| d_cust_hhi | -0.837 | TWIN |
| d_n_supp | 0.632 | no |
| log1p(a_in3) | 0.381 | no |
| d_supp_hhi | -0.379 | no |
| d_supp_top1 | -0.336 | no |
| d_tx_cp_share | 0.202 | no |


## 2 — Single-feature group-fold Y3

Y3 d_n_cust 0.653 n=3,003 pos=221 (peek 0.653 / 3,003 / 221 CONFIRM). vs size 0.617 vs days 0.711 vs top1 0.590 vs HHI 0.595. Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=0.036 PASS.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_n_cust | 3,003 | 221 | 0.653 | 0.650 | 0.077 | -1 | 0.742 0.662 0.530 0.665 0.666 |
| y3_recover_cash_6m | log1p(d_n_cust) | 3,003 | 221 | 0.653 | 0.650 | 0.077 | -1 | 0.742 0.662 0.530 0.665 0.666 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | d_cust_top1 | 2,485 | 141 | 0.590 | 0.590 | 0.116 | 1 | 0.738 0.411 0.588 0.607 0.606 |
| y3_recover_cash_6m | d_cust_hhi | 2,485 | 141 | 0.595 | 0.585 | 0.101 | 1 | 0.740 0.454 0.581 0.608 0.590 |
| y3_recover_cash_6m | d_n_supp | 3,003 | 221 | 0.699 | 0.685 | 0.116 | -1 | 0.820 0.722 0.514 0.764 0.674 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |


## 3 — Honest leftover after days

d_n_cust leftover after days OLS 0.648 rank 0.545 fake=False almost=True ρ(resid,days)=-0.469 R²=0.063 honest_dies=True n=3,003 pos=221. Inverse: days leftover after n_cust OLS 0.748 rank 0.699 dies=False. log1p leftover rank 0.545.

OLS folds: 0.605 0.685 0.605 0.635 0.708. Rank folds: 0.666 0.505 0.491 0.531 0.531.

## 4 — Twin / SIZE screen (in cut 1)

SIZE=False twin_gate=True twins=['d_cust_top1', 'd_cust_hhi'].

## 5 — Leftover after top1 / HHI / days+top1

n_cust leftover after top1 rank 0.456 dies=True; after HHI 0.459 dies=True; after days+top1 0.564 dies=False. Inverse top1 after n_cust 0.532; HHI after n_cust 0.532.

| bar | OLS | rank | ρ(resid,bar) | R2 | dies | n | n_pos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| after top1 | 0.590 | 0.456 | 0.523 | 0.183 | True | 2485 | 141 |
| after HHI | 0.596 | 0.459 | 0.454 | 0.157 | True | 2485 | 141 |
| after days+top1 | 0.678 | 0.564 | -0.264 | 0.198 | False | 2485 | 141 |
| after days+HHI | 0.678 | 0.566 | -0.274 | 0.172 | False | 2485 | 141 |
| top1 after n_cust | 0.599 | 0.532 | -0.649 | 0.183 | True | 2485 | 141 |
| HHI after n_cust | 0.608 | 0.532 | -0.685 | 0.157 | True | 2485 | 141 |


## 6 — Dark 470 stay NaN; ERP-only leftover

Dark 470 (want 470) n_cust nn=0 zero=0 CONFIRM NaN not 0. ERP Y3 0.653 leftover after days rank 0.545 dies=True n=3,003 pos=221.

| book | n_co | n_cm | n_cust nn | Y3 n | Y3 pos | CV | leftover rank | dies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dark | 470 | 7603 | 0 | 0 | 0 | LOW_POWER | — | — |
| ERP | 744 | 13554 | 11293 | 3003 | 221 | 0.653 | 0.545 | True |


## 7 — Q6 lag1 leftover after days_lag1

Y3 n_cust_lag1 0.635 leftover after days_lag1 rank 0.539 dies=True. lag3 leftover after days_lag3 0.517 dies=True. Days lag1 0.684 (quote 0.684 CONFIRM). Short Y3 lag1 leftover 0.552 n=1,544 pos=105. Short Y4 lag3 present 26.8% pos=56 (HHI Q6 quote 21.7% / 42 pos).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_n_cust | 3,003 | 221 | 0.653 | 0.650 | 0.077 | -1 | 0.742 0.662 0.530 0.665 0.666 |
| y3_recover_cash_6m | d_n_cust_lag1 | 2,804 | 209 | 0.635 | 0.631 | 0.060 | -1 | 0.706 0.657 0.541 0.623 0.647 |
| y3_recover_cash_6m | d_n_cust_lag3 | 2,274 | 171 | 0.604 | 0.600 | 0.064 | -1 | 0.673 0.624 0.517 0.559 0.644 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | 0.694 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |


## 8 — Y4 leftover after top1_lag3; monopoly tail

Y4 n_cust 0.566 lag3 0.578 vs top1_lag3 0.604 vs HHI_lag3 0.605. Leftover n_cust_lag3 after top1_lag3 rank 0.530 dies=True. HHI>0.975 Y4 rate 19.6% (quote 22.1% CONFIRM) vs rest 11.5% (quote 11.5%). n_cust==1 rate 19.2% vs ≥2 12.0%. Share of tail with n_cust==1 76.5% — not just n=1.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y4_ds_r_double | d_n_cust | 1,218 | 171 | 0.566 | 0.591 | 0.089 | -1 | 0.608 0.490 0.497 0.702 0.532 |
| y4_ds_r_double | d_n_cust_lag3 | 986 | 142 | 0.578 | 0.597 | 0.106 | -1 | 0.665 0.404 0.558 0.657 0.606 |
| y4_ds_r_double | d_cust_top1_lag3 | 848 | 116 | 0.604 | 0.590 | 0.043 | 1 | 0.666 0.613 0.588 0.547 0.604 |
| y4_ds_r_double | d_cust_hhi_lag3 | 848 | 116 | 0.605 | 0.592 | 0.044 | 1 | 0.677 0.610 0.575 0.563 0.598 |


| bar | rank | OLS | dies | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| n_cust_lag3 after top1_lag3 | 0.530 | 0.497 | True | 848 | 116 |
| n_cust_lag3 after HHI_lag3 | 0.525 | 0.495 | True | 848 | 116 |
| n_cust after top1 (now) | 0.468 | 0.425 | True | 1047 | 137 |


## 9 — vs `d_n_supp`

ρ(n_cust, n_supp)=0.632 different object. Y3 n_supp 0.699 leftover after days 0.587 dies=False. n_cust leftover after n_supp 0.535 dies=True. n_supp leftover after n_cust 0.649 dies=False.

## 10 — Holdout coverage only

Holdout 72 co / 1073 CM nn=542 cov=50.5% n0=40 p50=12.0 dark nn=0 (no fit, no AUROC).

## Extras

### ICC / demean

ICC=0.983 TRAIT k=744. Demean CV 0.527 leftover-days 0.548 dies=True. Company-mean CV 0.676 leftover-days 0.579 dies=False.

### n==0 / n==1 / ≥2

Drop n==0: Y3 0.596 leftover after days 0.530 dies=True n=2,485.

| bin | n | n_pos | Y3 rate |
| --- | --- | --- | --- |
| n==0 | 518 | 80 | 15.4% |
| n==1 | 363 | 37 | 10.2% |
| n 2-5 | 665 | 41 | 6.2% |
| n>=6 | 1457 | 63 | 4.3% |


### Fold-wise leftover; 12-name Y2 drop

Y2 n_cust 0.537 days 0.571. Drop 12 chronic: 0.537. Y2 leftover after days 0.446. Y3 leftover wo12 0.545 dies=True.

| fold | OLS leftover | rank leftover | n_va | n_pos |
| --- | --- | --- | --- | --- |
| 0 | 0.605 | 0.666 | 455 | 26 |
| 1 | 0.685 | 0.505 | 244 | 38 |
| 2 | 0.605 | 0.491 | 681 | 39 |
| 3 | 0.635 | 0.531 | 813 | 51 |
| 4 | 0.708 | 0.531 | 810 | 67 |


### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.427 p50=0.561 p95=0.613 share<0.55=40.0% n=50.

### Leftover by so-far

Leftover after days by so-far bucket.

| so-far | n | n_pos | raw | rank leftover | dies |
| --- | --- | --- | --- | --- | --- |
| <6 | 371 | 27 | — | — | False |
| 6-11 | 1366 | 90 | 0.658 | 0.526 | True |
| 12-17 | 1130 | 93 | 0.624 | 0.431 | True |
| 18-23 | 136 | 11 | — | — | False |
| 24+ | 0 | 0 | — | — | False |


### log1p vs raw

same-n raw 0.653 leftover 0.545; log1p 0.653 leftover 0.545. raw leftover after log1p 0.653 dies=False.

### Permute within days quintile

Permuted-within-days leftover rank p50=0.533 p90=0.553 n=30.

### leftover after a_n_tx

n_cust leftover after a_n_tx rank 0.545 dies=True. after days+a_n_tx 0.543 dies=True.

### leftover after size

n_cust leftover after size rank 0.618 dies=False. after days+size 0.544 dies=True. size leftover after n_cust 0.569 dies=False (in3 leftover after days was 0.521 — not overwritten).

### Without weakest / strongest fold

n_cust without weakest fold 2 = 0.684; without strongest 0 = 0.631. Rank leftover without fold 0 = 0.514 (still dies).

### leftover after days on top1-defined only

top1-defined same-n n=2,485 pos=141. raw 0.596 days 0.730. leftover after days 0.530 dies=True ρ(resid,days)=-0.469. after top1 0.456 dies=True. after days+top1 0.564 dies=False ρ(resid,days)=-0.264 R²=0.198. 0.564 is sample-shift (drop n=0), not a two-bar leftover.

### n==0 dummy leftover

n==0 dummy Y3 0.594 leftover after days 0.540 dies=True n=3,003 pos=221. n>0 leftover after days 0.530 dies=True raw 0.596. Zero pile is the 0.653, leftover still dies either way.

### monopoly Jaccard n==1 vs HHI>0.975

Y4-labeled n==1 ∩ HHI>0.975 = 156; Jaccard 76.5%. n==1 dummy Y4 0.520 leftover after top1 0.450 dies=True. Count rewrite of monopoly tail — leftover after top1 dies.

### SIZE tercile leftover

SIZE tercile leftover of n_cust after days. Inverse-size recover is days, not leftover count.

| tercile | n | n_pos | CV | leftover rank | dies | Y3 rate |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | 692 | 127 | 0.608 | 0.557 | False | 16.7% |
| T2 | 1138 | 59 | 0.433 | 0.426 | True | 5.6% |
| T3 | 1173 | 35 | LOW_POWER | — | False | 2.9% |


### company-mean leftover after days+top1

company-mean leftover after days 0.579 dies=False; after days+top1 0.450 dies=True. TRAIT leftover of who-has-customers is not a reason to KEEP as engine X.

### T1 leftover probe

T1 n=692 pos=127 raw 0.608 days 0.594. leftover after days 0.557 dies=False ρ(resid,days)=-0.469 folds=0.616 0.483 0.485 0.558 0.644. after days+top1 0.738 dies=False. after days+size 0.556 dies=False. T1 leftover lives after days+size.

### Y4 leftover after days

Y4 n_cust 0.566 leftover after days 0.544 dies=True; after top1 0.468 dies=True. Y4 leftover after days is not a KEEP path — monopoly bar is top1_lag3.

### Y3 rate by n_cust quintile

Y3 n_cust Q1→Q5 ['13.3%', '7.6%', '4.7%', '2.5%', '5.4%']; days Q1→Q5 ['19.4%', '7.5%', '3.3%', '3.8%', '2.0%']. Inverse gradient, days steeper.

| q | n_cust rate | n | n_pos |
| --- | --- | --- | --- |
| 1 | 13.3% | 881 | 117 |
| 2 | 7.6% | 445 | 34 |
| 3 | 4.7% | 516 | 24 |
| 4 | 2.5% | 564 | 14 |
| 5 | 5.4% | 597 | 32 |


### T1 ∩ top1 leftover

T1 Y3-labeled 1,331 / top1-defined 458 pos=60. raw 0.452 leftover after days 0.570 dies=False. after days+top1 0.738 dies=False folds=0.932 0.817 0.594 0.646 0.700. 0.738 is the top1-defined T1 slice — fold-noisy, not a KEEP pocket.

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| days_lag1 | 0.684 |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `d_n_cust` on the 15-col card.

## Files written

- `analysis/evaluate/n_cust_qa.py`
- `analysis/outputs/n_cust_qa.md`
- `analysis/outputs/n_cust_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_n_cust.md` (end, if WRITE_WAVE)

Elapsed 34s. Failed: none.

