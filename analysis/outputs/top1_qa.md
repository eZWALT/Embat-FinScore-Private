# Unused leftover of `d_cust_top1` after days as Y3 X

Generated `2026-09-19T06:05:08+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_top1`. Do not put top1 on the 15-col card. Do not overwrite `cust_hhi_qa.*`, `n_cust_qa.*`, `n_supp_qa.*`. Do not grow TURNOVER. Javier concentration is **top1**, not HHI.

`d_cust_top1` = share of AR invoice |amount| from the single largest customer (Family D, trailing 6m). Incomplete 6m books are NaN. Dark 470 stay NaN not 0. `d_cust_hhi` DROP as weaker rewrite (ρ 0.994). `d_n_cust` CLOSED leftover 0.545.

## Headline

`d_cust_top1` as Y3 X: **CLOSE unused leftover** (unused leftover after days: honest rank 0.525 dies (OLS 0.419 fake=False). Also TWIN of ['d_cust_hhi', 'd_n_cust']. DROP from the 44 as Y3 X. Y4 >0.975 footnote KEEP. Do not invent y_top1. Off the 15-col card.). Y3 leftover after days OLS 0.419 rank 0.525 (dies, fake=False). Inverse days after top1 rank 0.715. Single 0.590 vs size 0.617 vs days 0.711 vs HHI 0.595 vs n_cust 0.653. after HHI 0.434 after n_cust 0.532 after days+HHI 0.572. Y4 footnote **KEEP**. 15-col card: no — do not put d_cust_top1 on the 15-col card. PARK as Y — do not invent y_top1. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — do not invent y_top1. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Q6 lag1 leftover after days_lag1 0.523. |
| 3 | Who is turning? | **CLOSE unused leftover** leftover after days 0.525 vs days 0.711. |
| 4 | Dip vs fall? | Y4 >0.975 footnote **KEEP** lag3 0.604. |
| 5 | Why did it change? | Twin screen: ['d_cust_hhi', 'd_n_cust']. ρ vs HHI 0.994. |
| 6 | Months earlier? | lag1 leftover 0.523; days_lag1 0.684. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `d_cust_top1` as Y3 X / the 15-col card | **CLOSE unused leftover** | unused leftover after days: honest rank 0.525 dies (OLS 0.419 fake=False). Also TWIN of ['d_cust_hhi', 'd_n_cust']. DROP from the 44 as Y3 X. Y4 >0.975 footnote KEEP. Do not invent y_top1. Off the 15-col card. |
| `d_cust_top1` as engine X on the 44 | **DROP** | leftover lives=False twin=True SIZE=False beat-size=False |
| Y4 >0.975 monopoly footnote | **KEEP** | lag3 0.604; HHI after top1_lag3 OLS 0.549 / rank 0.461; now-tail 19.6% vs 11.5% |
| `y_top1` | **PARK** | do not invent a concentration Y |
| twin of HHI / n_cust | 0.994 / -0.806 | leftover after HHI 0.434 after n_cust 0.532 |
| same object as `d_supp_top1` | **NO** | ρ=0.117 |
| Q6 lag1 after days_lag1 | **CLOSE** | leftover 0.523 |

## 1 — Coverage; twin / SIZE screen

Train 1,214 co / 21,157 CM. d_cust_top1 cov 42.2% acf1=0.737. Dark 470 (want 470) nn=0 zero=0 CONFIRM NaN. ρ vs days -0.287 vs a_n_tx -0.305 vs HHI 0.994 (peek 0.994 CONFIRM) vs n_cust -0.806 (peek −0.806 CONFIRM) vs size -0.169. SIZE=False twins=['d_cust_hhi', 'd_n_cust'] twin_gate=True.

| col | n_nn | cov | acf1 |
| --- | --- | --- | --- |
| d_cust_top1 | 8,928 | 42.2% | 0.737 |
| d_cust_hhi | 8,928 | 42.2% | 0.768 |
| d_n_cust | 11,293 | 53.4% | 0.867 |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | -0.287 | no |
| a_n_tx | -0.305 | no |
| d_cust_hhi | 0.994 | TWIN |
| d_n_cust | -0.806 | TWIN |
| log1p(a_in3) | -0.169 | no |
| d_supp_top1 | 0.117 | no |
| d_n_supp | -0.304 | no |
| d_tx_cp_share | -0.112 | no |


## 2 — Single-feature group-fold Y3

Y3 d_cust_top1 0.590 n=2,485 pos=141 (peek 0.590 / 2,485 / 141 CONFIRM). vs size 0.617 vs days 0.711 vs HHI 0.595 vs n_cust 0.653. Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=-0.027 FAIL.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_cust_top1 | 2,485 | 141 | 0.590 | 0.590 | 0.116 | 1 | 0.738 0.411 0.588 0.607 0.606 |
| y3_recover_cash_6m | d_cust_hhi | 2,485 | 141 | 0.595 | 0.585 | 0.101 | 1 | 0.740 0.454 0.581 0.608 0.590 |
| y3_recover_cash_6m | d_n_cust | 3,003 | 221 | 0.653 | 0.650 | 0.077 | -1 | 0.742 0.662 0.530 0.665 0.666 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | d_supp_top1 | 2,877 | 203 | 0.641 | 0.605 | 0.102 | 1 | 0.787 0.702 0.561 0.613 0.542 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |


## 3 — Honest leftover after days

d_cust_top1 leftover after days OLS 0.419 rank 0.525 fake=False almost=False ρ(resid,days)=0.000 R²=0.067 honest_dies=True n=2,485 pos=141. Inverse: days leftover after top1 OLS 0.716 rank 0.715 dies=False.

OLS folds: 0.329 0.336 0.428 0.523 0.479. Rank folds: 0.671 0.349 0.582 0.520 0.501.

## 4 — Twin / SIZE screen (in cut 1)

SIZE=False twin_gate=True twins=['d_cust_hhi', 'd_n_cust'].

## 5 — Leftover after HHI / n_cust / days+HHI

top1 leftover after HHI rank 0.434 dies=True (want die — HHI is the rewrite). after n_cust 0.532 dies=True; after days+HHI 0.572 dies=False. Inverse HHI after top1 0.436 (cust_hhi leftover after top1 was 0.549 — not overwritten). n_cust after top1 0.456 (n_cust leftover 0.545 — not overwritten).

| bar | OLS | rank | ρ(resid,bar) | R2 | dies | n | n_pos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| after HHI | 0.553 | 0.434 | 0.017 | 0.966 | True | 2485 | 141 |
| after n_cust | 0.599 | 0.532 | -0.649 | 0.183 | True | 2485 | 141 |
| after days+HHI | 0.431 | 0.572 | -0.015 | 0.966 | False | 2485 | 141 |
| after days+n_cust | 0.545 | 0.565 | 0.013 | 0.211 | False | 2485 | 141 |
| HHI after top1 | 0.563 | 0.436 | 0.092 | 0.966 | True | 2485 | 141 |
| n_cust after top1 | 0.590 | 0.456 | 0.523 | 0.183 | True | 2485 | 141 |


## 6 — Dark 470 stay NaN; ERP leftover

Dark 470 (want 470) top1 nn=0 zero=0 CONFIRM NaN not 0. ERP Y3 0.590 leftover after days 0.525 dies=True.

| book | n_co | n_cust_top1 nn | Y3 n | Y3 pos | CV |
| --- | --- | --- | --- | --- | --- |
| dark | 470 | 0 | 0 | 0 | LOW_POWER |
| ERP | 744 | 8928 | 2485 | 141 | 0.590 |


## 7 — Q6 lag1 leftover after days_lag1

Y3 top1_lag1 0.576 leftover after days_lag1 rank 0.523 dies=True. lag3 leftover after days_lag3 0.452 dies=True. Days lag1 0.684 (quote 0.684 CONFIRM). Short Y4 lag3 present 21.7% pos=42 (HHI Q6 quote 21.7% / 42 pos).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_cust_top1 | 2,485 | 141 | 0.590 | 0.590 | 0.116 | 1 | 0.738 0.411 0.588 0.607 0.606 |
| y3_recover_cash_6m | d_cust_top1_lag1 | 2,309 | 135 | 0.576 | 0.577 | 0.110 | 1 | 0.700 0.401 0.563 0.605 0.609 |
| y3_recover_cash_6m | d_cust_top1_lag3 | 1,862 | 115 | 0.541 | 0.536 | 0.101 | 1 | 0.646 0.405 0.483 0.538 0.632 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | 0.694 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |


## 8 — Body vs tail as Y3 X; Y4 footnote

Y3 body leftover after days 0.398 dies=True raw 0.352. Y3 tail raw — n=502 pos=47. Y4 top1_lag3 0.604 HHI_lag3 0.605 (quote 0.605 CONFIRM). HHI leftover after top1_lag3 OLS 0.549 (quote 0.549 CONFIRM) rank 0.461 (cust_hhi rank-ortho 0.461 CONFIRM). Y4 contemporaneous tail rate 19.6% vs rest 11.5% (lag3 quote 22.1% / 11.5% is HHI_lag3, not now-HHI). Y4 footnote KEEP.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | top1 body HHI≤0.975 | 1,983 | 94 | 0.352 | 0.533 | 0.141 | 1 | 0.154 0.276 0.503 0.367 0.460 |
| y3_recover_cash_6m | top1 tail HHI>0.975 | 502 | 47 | LOW_POWER | LOW_POWER | — | — | — |
| y4_ds_r_double | d_cust_top1 | 1,047 | 137 | 0.583 | 0.573 | 0.072 | 1 | 0.613 0.648 0.525 0.640 0.488 |
| y4_ds_r_double | d_cust_top1_lag3 | 848 | 116 | 0.604 | 0.590 | 0.043 | 1 | 0.666 0.613 0.588 0.547 0.604 |
| y4_ds_r_double | d_cust_hhi_lag3 | 848 | 116 | 0.605 | 0.592 | 0.044 | 1 | 0.677 0.610 0.575 0.563 0.598 |
| y4_ds_r_double | top1_lag3 body | 665 | 74 | 0.612 | 0.567 | 0.103 | 1 | 0.565 0.780 0.640 0.552 0.523 |


## 9 — vs `d_supp_top1`

ρ(cust_top1, supp_top1)=0.117 different object. Y3 supp_top1 0.641 leftover after days 0.429 dies=True. cust_top1 leftover after supp_top1 0.587 dies=False.

## 10 — Holdout coverage only

Holdout 72 co / 1073 CM nn=502 cov=46.8% p50=0.489 dark nn=0 (no fit, no AUROC).

## Extras

### ICC / demean

ICC=0.981 TRAIT k=674. Demean CV 0.409 leftover-days 0.404 dies=True. Company-mean CV 0.613 leftover-days 0.539 dies=True.

### Fold-wise leftover; 12-name Y2 drop

Y2 top1 0.575. Drop 12 chronic. Y3 leftover wo12 0.525 dies=True.

| fold | OLS leftover | rank leftover | n_va | n_pos |
| --- | --- | --- | --- | --- |
| 0 | 0.329 | 0.671 | 401 | 20 |
| 1 | 0.336 | 0.349 | 188 | 19 |
| 2 | 0.428 | 0.582 | 544 | 30 |
| 3 | 0.523 | 0.520 | 709 | 34 |
| 4 | 0.479 | 0.501 | 643 | 38 |


### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.394 p50=0.538 p95=0.599 share<0.55=65.0% n=40.

### Permute within days quintile

Permuted-within-days leftover rank p50=0.521 p90=0.563 n=24.

### leftover after size

top1 leftover after size rank 0.577 dies=False. after days+size 0.523 dies=True.

### Y3 rate by top1 quintile

Y3 top1 Q1→Q5 ['4.8%', '3.4%', '4.4%', '6.2%', '9.5%'].

| q | top1 rate | n | n_pos |
| --- | --- | --- | --- |
| 1 | 4.8% | 497 | 24 |
| 2 | 3.4% | 497 | 17 |
| 3 | 4.4% | 497 | 22 |
| 4 | 6.2% | 497 | 31 |
| 5 | 9.5% | 497 | 47 |


### SIZE tercile leftover

SIZE tercile leftover of top1 after days.

| tercile | n | n_pos | CV | leftover rank | dies |
| --- | --- | --- | --- | --- | --- |
| T1 | 458 | 60 | 0.463 | 0.437 | True |
| T2 | 970 | 56 | 0.587 | 0.549 | True |
| T3 | 1057 | 25 | LOW_POWER | — | False |


### HHI leftover reconcile (OLS 0.549 vs rank 0.461)

cust_hhi 0.549 is OLS leftover of HHI after top1_lag3 0.549 CONFIRM; honest rank 0.461 CONFIRM 0.461 R²=0.966. Y4 HHI_lag3 >0.975 22.1% n=172 pos=38 vs rest 11.5% CONFIRM 22.1/11.5. HHI_lag3 body CV 0.445 (quote 0.445 CONFIRM). Y3 tail-flag leftover after days 0.461 dies=True raw 0.521.

| cut | value | quote | ok |
| --- | --- | --- | --- |
| Y4 HHI after top1_lag3 OLS | 0.549 | 0.549 | CONFIRM |
| Y4 HHI after top1_lag3 rank | 0.461 | 0.461 | CONFIRM |
| Y4 HHI_lag3 >0.975 rate | 22.1% | 22.1% | CONFIRM |
| Y4 HHI_lag3 rest rate | 11.5% | 11.5% | CONFIRM |
| Y4 HHI_lag3 body CV | 0.445 | 0.445 | CONFIRM |


### days+HHI leftover is rewrite residual

after days rank 0.525 n=2,485; after days+HHI rank 0.572 n=2485 R²=0.966 same_n=True. LIVES is rewrite residual (R²≥0.90), not leftover. top1-after-HHI residual leftover after days 0.456 dies=True R²_on_HHI=0.966. top1 leftover after a_n_tx 0.529 dies=True.

### Y3 body vs tail rates

Y3 tail HHI>0.975 rate 9.4% n=502 pos=47 vs body 4.7% n=1983 pos=94. Q5 leftover after days — dies=False n=497 pos=47.

### Q6 same-n; incomplete 6m; Y4 leftover after days

Same-n lag1 rows: contemporaneous leftover after days 0.413 dies=True; lag1 leftover after days_lag1 0.523 dies=True n=2309. Calendar incomplete (period<2025-02) top1 nn=0 CONFIRM NaN. Company-relative so-far<6 nn=1310 (wrong clock — not the D NaN). Y4 leftover of top1 after days 0.559 dies=False; top1_lag3 after days_lag3 0.567 dies=False (report-only; do not promote top1 as Y4 engine X).

### ERP trail-length leftover

ERP trail-length leftover of top1 after days.

| slice | n | n_pos | CV | leftover | dies |
| --- | --- | --- | --- | --- | --- |
| ERP short_<12 | 1435 | 71 | 0.596 | 0.531 | True |
| ERP mid_12_17 | 939 | 64 | 0.578 | 0.529 | True |
| ERP long_>=18 | 111 | 6 | LOW_POWER | — | False |


### Drop fold 1

Drop fold 1: Y3 top1 0.635 leftover after days 0.568 dies=False n=2297 pos=122.

### Y4 leftover of top1 (report-only)

Y4 top1 0.583 leftover after days 0.559 dies=False; after HHI 0.444 dies=True R²=0.966. Y4 top1_lag3 0.604 leftover after days_lag3 0.567 dies=False; after HHI_lag3 0.506 dies=True. Do not put top1 on Y4 engine — footnote is the >0.975 tail, not leftover as X.

### Drop-one-fold leftover

Drop-one-fold leftover lives on 1/5 drops. Fold instability is not KEEP — full-sample leftover dies and twin_gate stays.

| drop fold | n | n_pos | CV | leftover | dies | beat-size |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2084 | 121 | 0.553 | 0.437 | True | False |
| 1 | 2297 | 122 | 0.635 | 0.568 | False | False |
| 2 | 1941 | 111 | 0.591 | 0.425 | True | False |
| 3 | 1776 | 107 | 0.586 | 0.441 | True | False |
| 4 | 1842 | 103 | 0.586 | 0.530 | True | False |


### Javier top1≥0.90 flag; leftover inside days bins

top1≥0.90 flag Y3 0.575 leftover after days 0.463 dies=True n=2485 pos=141. Inside days quintiles top1 is a rate gradient, not leftover after days.

| days q | n | n_pos | CV | rate |
| --- | --- | --- | --- | --- |
| 1 | 543 | 75 | 0.583 | 13.8% |
| 2 | 539 | 28 | LOW_POWER | 5.2% |
| 3 | 490 | 16 | LOW_POWER | 3.3% |
| 4 | 483 | 18 | LOW_POWER | 3.7% |
| 5 | 430 | 4 | LOW_POWER | 0.9% |


## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| Y4 HHI_lag3 footnote | 0.605 |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `d_cust_top1` on the 15-col card.

## Files written

- `analysis/evaluate/top1_qa.py`
- `analysis/outputs/top1_qa.md`
- `analysis/outputs/top1_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_top1.md` (end, if WRITE_WAVE)

Elapsed 31s. Failed: none.

