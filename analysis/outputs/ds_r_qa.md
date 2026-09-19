# Unused leftover of KEEP-flow `f_ds_r` after days

Generated `2026-09-19T05:25:32+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_ds_r`. Do not score Family F vs Y4 as KEEP. Do not grow TURNOVER. Do not change the 15-col Y3 card.

`f_ds_r` = trailing-3m debt service / inflow, clipped [0, 2] (Family F). Still on the 15-col Y3 card and the 44. Contemporaneous `f_fc_r` was DROPPED from the 44 as unused leftover (ρ vs `f_ds_r` 0.205). This cut asks leftover after `c_n_days_with_tx` (0.711 bar).

## Headline

`f_ds_r` on the 15-col card: **DROP** (unused leftover after days: honest rank 0.528 dies (OLS 0.709 ρ(resid,days)=0.775 fake=False). Single 0.620 loses to days 0.711 and fails beat-size (Δ=0.003). Twin of euro ds (ρ a_debt=0.881). i_lift drop-ds_r-only still 0.7525. Parent 15-col card absorbs; do not edit the card. Same unused leftover as contemp f_fc_r on the 44.). Y3 leftover after days OLS 0.709 rank 0.528 (dies, fake=False). Inverse days after ds_r rank 0.682. After size 0.567; after fc 0.606 (CONFIRM 0.606). Single 0.620 vs days 0.711 vs size 0.617 beat_size=False. ρ vs fc 0.205. Y4 identity only (not KEEP). TURNDSSWAP 0.712 not a swap. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | `f_ds_r` is a financing **flow**, already a card stem. DROP as unused leftover after days. |
| 2 | Who is improving? | Not a path column by itself. Q6 lag3 already CLOSE. |
| 3 | Who is turning? | Y4 *is* `y4_ds_r_double`. F forbidden as Y4 X. identity only — Y4 is y4_ds_r_double; F forbidden as X; not KEEP. |
| 4 | Dip vs fall? | TURNDSSWAP already 0.712 — not a swap. Do not refit TURNOVER. Do not rip `f_fc_r_lag3`. |
| 5 | Why did it change? | Leftover after days rank 0.528. After fc 0.606 still lives. Twin vs euro-ds: ['a_debt_service', 'f_debt_service']. |
| 6 | Months earlier? | lag3 empty until so-far≥6: CONFIRM. F lag3 **CLOSE**. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `f_ds_r` on the 15-col card | **DROP** | unused leftover after days: honest rank 0.528 dies (OLS 0.709 ρ(resid,days)=0.775 fake=False). Single 0.620 loses to days 0.711 and fails beat-size (Δ=0.003). Twin of euro ds (ρ a_debt=0.881). i_lift drop-ds_r-only still 0.7525. Parent 15-col card absorbs; do not edit the card. Same unused leftover as contemp f_fc_r on the 44. |
| `f_ds_r` on the 44 | **DROP** | same leftover test; parent absorbs; do not edit the card |
| F lag3 Q6 | **CLOSE** | empty until so-far≥6 already CLOSE |
| Y4 identity | **not KEEP** | identity only — Y4 is y4_ds_r_double; F forbidden as X; not KEEP |
| TURNOVER swap | **not a swap** | not a swap — TURNDSSWAP 0.712; do not refit; do not rip f_fc_r_lag3 |

## 1 — Coverage; 470 vs ERP

Train 1,214 co / 21,157 CM. f_ds_r cov 88.5% vs f_fc_r 88.5% SAME rolling-3 hole. Dark 470 co have ds (access ≠ ERP, same as fc). share=0 is no-repayment 3m, not a fill.

| col | n_nn | cov CM | ever co | ever % | ever >0 | share>0 | p50 | share=0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f_ds_r | 18,729 | 88.5% | 1,214 | 100.0% | 491 | 28.8% | 0.0000 | 71.2% |
| f_fc_r | 18,729 | 88.5% | 1,214 | 100.0% | 1,026 | 71.2% | 0.0003 | 28.8% |


| book | col | n_co | n_cm | n_nn | cov CM | ever co | share=0 | zero_fill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dark | f_ds_r | 470 | 7603 | 6663 | 87.6% | 470 | 68.1% | no — 0 is no-ds 3m, not a dark fill |
| dark | f_fc_r | 470 | 7603 | 6663 | 87.6% | 470 | 32.9% | no — 0 is no-ds 3m, not a dark fill |
| ERP | f_ds_r | 744 | 13554 | 12066 | 89.0% | 744 | 73.0% | no — 0 is no-ds 3m, not a dark fill |
| ERP | f_fc_r | 744 | 13554 | 12066 | 89.0% | 744 | 26.5% | no — 0 is no-ds 3m, not a dark fill |


Holdout coverage only: 72 co / 1073 CM nn=929 cov=86.6% share=0=66.5%.

## 2 — Spearman twins

ρ vs f_fc_r 0.205 CONFIRM ~0.205. vs a_debt_service 0.881 vs f_debt_service 0.881 vs days 0.335 vs log_in3 0.274 SIZE=False. Twins (≥0.80): ['a_debt_service', 'f_debt_service'].

| vs | rho | twin |
| --- | --- | --- |
| f_fc_r | 0.205 | no |
| a_debt_service | 0.881 | yes |
| f_debt_service | 0.881 | yes |
| a_fin_cost | 0.291 | no |
| c_n_days_with_tx | 0.335 | no |
| log_in3 | 0.274 | no |


## 3 — Single-feature group-fold AUROC

Y3 f_ds_r 0.620 vs days 0.711 vs size 0.617. Replica days 0.711 CONFIRM / size 0.617 CONFIRM / own 0.620 CONFIRM. Beat size ≥0.02: False (Δ=0.003).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | f_ds_r | 5,528 | 391 | 0.620 | 0.625 | 0.045 | -1 | 0.547 0.633 0.671 0.618 0.627 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log_in3 | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | f_fc_r | 5,528 | 391 | 0.559 | 0.541 | 0.084 | -1 | 0.550 0.541 0.703 0.500 0.500 |
| y3_recover_cash_6m | a_debt_service | 5,648 | 402 | 0.613 | 0.618 | 0.034 | -1 | 0.570 0.643 0.650 0.591 0.613 |
| y3_recover_cash_6m | f_debt_service | 5,648 | 402 | 0.613 | 0.618 | 0.034 | -1 | 0.570 0.643 0.650 0.591 0.613 |
| y2_neg_2of3 | f_ds_r | 14,968 | 1,044 | 0.538 | 0.539 | 0.066 | 1 | 0.539 0.621 0.438 0.532 0.557 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.577 | 0.046 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | log_in3 | 14,968 | 1,044 | 0.552 | 0.540 | 0.046 | 1 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | f_fc_r | 14,968 | 1,044 | 0.534 | 0.521 | 0.053 | 1 | 0.545 0.618 0.481 0.497 0.528 |
| y2_neg_2of3 | a_debt_service | 17,356 | 1,271 | 0.526 | 0.527 | 0.053 | 1 | 0.530 0.596 0.453 0.500 0.551 |
| y2_neg_2of3 | f_debt_service | 17,356 | 1,271 | 0.526 | 0.527 | 0.053 | 1 | 0.530 0.596 0.453 0.500 0.551 |
| y7_top1_lost | f_ds_r | 7,000 | 1,987 | 0.521 | 0.523 | 0.027 | -1 | 0.549 0.502 0.514 0.491 0.551 |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.511 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 |
| y7_top1_lost | log_in3 | 7,000 | 1,987 | 0.469 | 0.536 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 |
| y7_top1_lost | f_fc_r | 7,000 | 1,987 | 0.555 | 0.550 | 0.076 | 1 | 0.559 0.674 0.561 0.499 0.482 |
| y7_top1_lost | a_debt_service | 7,464 | 2,149 | 0.523 | 0.526 | 0.031 | -1 | 0.545 0.504 0.509 0.492 0.565 |
| y7_top1_lost | f_debt_service | 7,464 | 2,149 | 0.523 | 0.526 | 0.031 | -1 | 0.545 0.504 0.509 0.492 0.565 |


## 4 — Honest leftover after days

ds_r leftover after days OLS 0.709 rank 0.528 fake=False ρ(resid,days)=0.775 R²=0.001 honest_dies=True. Inverse: days leftover after ds_r OLS 0.720 rank 0.682 dies=False.

OLS folds: 0.619 0.725 0.732 0.745 0.723. Rank folds: 0.473 0.537 0.597 0.505 0.527.

## 5 — Leftover after size / after fc

after size OLS 0.666 rank 0.567 dies=False R²=0.005. after fc OLS 0.597 rank 0.606 CONFIRM ~0.606 dies=False R²=0.005.

## 6 — SIZE terciles; ICC / demean

ICC=0.948 TRAIT k=1214. Demean CV 0.452 company-mean 0.621. Demean leftover after days rank 0.545 dies=True.

| tercile | n | n_pos | CV | folds |
| --- | --- | --- | --- | --- |
| T1 | 1,298 | 215 | 0.549 | 0.517 0.538 0.580 0.541 0.569 |
| T2 | 2,043 | 115 | 0.557 | 0.531 0.683 0.473 0.561 0.536 |
| T3 | 2,187 | 61 | 0.660 | 0.489 0.683 0.690 0.665 0.772 |


## 7 — Q6 lag1 / lag3

lag3 empty until so-far≥6: train 0/1436 CONFIRM. Y3 lag1 0.617 lag3 0.604. F lag3 already CLOSE.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | f_ds_r | 5,528 | 391 | 0.620 | 0.625 | 0.045 | -1 | 0.547 0.633 0.671 0.618 0.627 |
| y3_recover_cash_6m | f_ds_r_lag1 | 5,078 | 355 | 0.617 | 0.623 | 0.055 | -1 | 0.530 0.630 0.684 0.617 0.623 |
| y3_recover_cash_6m | f_ds_r_lag3 | 4,212 | 313 | 0.604 | 0.612 | 0.064 | -1 | 0.507 0.638 0.665 0.576 0.636 |


| so-far | n_lab | ds nn | lag1 nn | lag3 nn |
| --- | --- | --- | --- | --- |
| <6 | 1436 | 1316 | 866 | 0 |
| 6-11 | 2287 | 2287 | 2287 | 2287 |
| 12-17 | 1712 | 1712 | 1712 | 1712 |
| 18-23 | 213 | 213 | 213 | 213 |
| 24+ | 0 | 0 | 0 | 0 |


## 8 — Y4 identity (not KEEP)

Y4 *is* y4_ds_r_double. Train labeled 2,370 pos=329. ρ(f_ds_r, Y4)=-0.082 Jaccard(ds≥0.5, Y4+)=0.068. leakage_check f_ds_r as Y4 X ok=False issues=["forbidden prefix 'f_': ['f_ds_r']"]. Not scored as KEEP. Family F forbidden as Y4 X.

## 10 — Fold-wise leftover; 12-name Y2 drop

Y2 f_ds_r 0.538 days 0.571. Drop 12 chronic names: ds 0.522 days 0.549. Y3 leftover after days without the 12: rank 0.527 dies=True.

| fold | OLS leftover | rank leftover | n_va | n_pos |
| --- | --- | --- | --- | --- |
| 0 | 0.619 | 0.473 | 1288 | 53 |
| 1 | 0.725 | 0.537 | 685 | 92 |
| 2 | 0.732 | 0.597 | 1047 | 64 |
| 3 | 0.745 | 0.505 | 1342 | 78 |
| 4 | 0.723 | 0.527 | 1166 | 104 |


| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | f_ds_r | 14,968 | 1,044 | 0.538 | 0.539 | 0.066 | 1 | 0.539 0.621 0.438 0.532 0.557 |
| y2_neg_2of3 | days | 17,356 | 1,271 | 0.571 | 0.577 | 0.046 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | f_ds_r wo12 | 14,776 | 881 | 0.522 | 0.518 | 0.074 | 1 | 0.460 0.621 0.438 0.532 0.557 |
| y2_neg_2of3 | days wo12 | 17,140 | 1,094 | 0.549 | 0.551 | 0.041 | 1 | 0.513 0.539 0.612 0.565 0.517 |


## Extras

### log1p intensity

log1p(f_ds_r) Y3 0.620 leftover after days OLS 0.526 rank 0.528 dies=True.

### ds vs fc stack leftover

after days+fc rank 0.536 dies=True R²=0.006. after days+size rank 0.529 dies=True. after a_debt_service rank 0.545 dies=True R²=0.010 (ratio vs euro).

### Javier debt_serv_r

f_ds_r vs roll3(a_debt_service)/max(a_in3,1) ρ=1.000 SAME. vs roll3(f_debt_service)/max(a_in3,1) ρ=1.000 SAME. Javier debt_serv_r is the same formula — SAME to F-trail.

### Holdout coverage; dark vs ERP

Holdout coverage 86.6% only — no fit. Dark Y3 0.610 leftover-days rank 0.521; ERP Y3 0.631 leftover-days rank 0.464.

### ds>0 months

ds>0 Y3 0.444 leftover after days rank 0.599 dies=False n=2206 pos=66.

### ACF / weakest fold

median ACF1=0.622 n_co=476. Y3 without weakest fold 0 = 0.638.

### Quintiles

Y3 rate by quintile — days monotone, ds_r flatter then T3 bump.

| feat | Q | n | n_pos | rate | med |
| --- | --- | --- | --- | --- | --- |
| f_ds_r | 1 | 4422 | 361 | 8.2% | 0.0000 |
| f_ds_r | 2 | 1106 | 30 | 2.7% | 0.1981 |
| days | 1 | 1235 | 210 | 17.0% | 4.0000 |
| days | 2 | 1171 | 85 | 7.3% | 12.0000 |
| days | 3 | 1160 | 52 | 4.5% | 19.0000 |
| days | 4 | 1107 | 32 | 2.9% | 22.0000 |
| days | 5 | 975 | 23 | 2.4% | 28.0000 |


### Leftover by SIZE tercile / so-far

Company-mean leftover after days rank 0.555 dies=False. Euro a_debt_service leftover after days rank 0.484 dies=True. Days leftover after size rank 0.667 dies=False (days is not SIZE).

| slice | n | n_pos | rank | ols | dies |
| --- | --- | --- | --- | --- | --- |
| size T1 | 1298 | 215 | 0.517 | 0.610 | True |
| size T2 | 2043 | 115 | 0.462 | 0.598 | True |
| size T3 | 2187 | 61 | 0.634 | 0.675 | False |
| so-far <6 | 1316 | 78 | 0.528 | 0.687 | True |
| so-far 6-11 | 2287 | 163 | 0.524 | 0.727 | True |
| so-far 12-17 | 1712 | 134 | 0.561 | 0.650 | False |
| so-far 18-23 | 213 | 16 | — | — | False |
| so-far 24+ | 0 | 0 | — | — | False |


### i_lift drop-ds_r-only (quoted, no new GBM)

i_lift.md: drop `f_ds_r` only (no I) 12-col still **0.7525** (Δ 0.000 vs 0.752). Parent 15-col card absorbs. Do not edit the card. Do not merge I. `i_io_x_dsr` ρ=0.980 / `i_dso_x_dsr` ρ=0.916 vs `f_ds_r` — I twins, not new leftover.

### Fold-0 leftover / OLS fake-days leak

Rank leftover fold 0=0.473 without fold 0=0.541. OLS leftover 0.709 with ρ(resid,days)=0.775 almost-fake days leak (rank is honest). Ticket: OLS can fake a days leak; leftover <0.55 dies.

### Company bootstrap leftover after days

Bootstrap leftover-after-days rank p50=0.534 p10=0.458 p90=0.559 share<0.55=78.8% n=80.

### ds>0 quintiles

share=0 pile is Q1 of the full qcut (8.2% Y3). ds>0 quintiles sit on a thinner positive tail.

| Q | n | n_pos | rate | med ds | med days |
| --- | --- | --- | --- | --- | --- |
| 1 | 442 | 16 | 3.6% | 0.0017 | 22.000 |
| 2 | 441 | 18 | 4.1% | 0.0185 | 23.000 |
| 3 | 441 | 3 | 0.7% | 0.0616 | 22.000 |
| 4 | 441 | 12 | 2.7% | 0.1569 | 22.000 |
| 5 | 441 | 17 | 3.9% | 0.5295 | 17.000 |
| share=0 | 3322 | 325 | 9.8% | 0 | 14.000 |


### T3 leftover leak check

T3 n=2187 pos=61 raw 0.660 days 0.624. leftover after days OLS 0.675 rank 0.634 ρ(resid,days)=0.775 fake=False dies=False. after days+size rank 0.649 dies=False. T3 leftover lives — still not KEEP on the full card.

### Permute within days quintile

Permuted-within-days leftover rank p50=0.535 p90=0.552 n=40. Observed leftover 0.528 should sit inside this null if unused.

### Ever-ds>0 companies

Ever-ds>0 cos=491 labeled n=2949 pos=123 raw 0.590 leftover-days rank 0.534 dies=True.

### Binary has-ds>0 leftover

has-ds>0 Y3 0.617 leftover after days rank 0.521 dies=True ρ vs days=0.373. The 0.620 single is mostly the zero-pile vs any-repayment, which days already owns.

### T3 leftover folds

T3 leftover folds OLS 0.508 0.695 0.694 0.770 0.709 rank 0.458 0.672 0.679 0.563 0.801. Rank 0.634 lives on the large-book slice only. Full-card leftover 0.528 dies. Not KEEP.

### Javier score_pipeline recon

score_pipeline monthly-flow debt_serv_r vs store f_ds_r ρ=0.999 SAME n=20,268 train rows (no invoice join).

### Y7 leftover after issued

Y7 f_ds_r 0.521 leftover after e_ar_issued rank 0.556 dies=False. TURNDSSWAP already 0.712 — not a swap. Do not refit TURNOVER.

### Y2 leftover after days

Y2 leftover after days rank 0.449 dies=True. Without 12 chronic names 0.458 dies=True (days wo12 already 0.549).

### Leftover by company trail length

Leftover after days by company trail length.

| book | n | n_pos | raw | rank leftover | dies |
| --- | --- | --- | --- | --- | --- |
| short_<12 | 118 | 7 | — | — | False |
| mid_12_17 | 348 | 33 | — | — | False |
| long_>=18 | 5062 | 351 | 0.615 | 0.526 | True |


### TURNDSSWAP

Already **0.712** — not a swap for `f_fc_r_lag3`. Do not refit TURNOVER. Do not rip `f_fc_r_lag3` off TURNOVER. Night Y7 stays **0.720 / 0.712**.

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| TURNDSSWAP | 0.712 not a swap |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put new cols on the 15-col card.

## Files written

- `analysis/evaluate/ds_r_qa.py`
- `analysis/outputs/ds_r_qa.md`
- `analysis/outputs/ds_r_leftover.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_ds_r.md` (end, if WRITE_WAVE)

Elapsed 50s. Failed: none.

