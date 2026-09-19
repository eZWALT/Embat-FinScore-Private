# Unused leftover of `h_sib_neg_share` after days as Y3 X

Generated `2026-09-19T06:16:05+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_sib_neg`. Do not put H on the 15-col card. Do not overwrite `sibling_h.*`. Y3 never B. Hidden test is new groups. Do not grow TURNOVER.

`h_sib_neg_share` = (# siblings with net < 0) / (h_group_size − 1) (Family H). Family H already PARK as Y3 X (sibling_h). This lane is leftover after days on the 44 stem. Sister mean `b_runway` ≥ 1 is a Q5 footnote only.

## Headline

`h_sib_neg_share` as Y3 X: **CLOSE unused leftover** (unused leftover after days: honest rank 0.453 dies (OLS 0.453 fake=False). Single 0.434 vs days 0.711 / size 0.617. DROP from the 44 as Y3 X. PARK as Y (H marks sister existence; hidden test is new groups). Q5 sister-runway footnote KEEP.). Y3 leftover after days OLS 0.453 rank 0.453 (dies, fake=False). Inverse days after sib_neg rank 0.724. Single 0.434 vs size 0.617 vs days 0.711 vs share 0.638 vs mixed 0.525. after size 0.446 after mixed 0.435. Q5 sister-runway footnote **KEEP**. 15-col card: no — do not put H on the 15-col card. PARK as Y — H is sister existence, not health. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — H is sister existence, not health. Hidden test is new groups. |
| 2 | Who is improving? | Q6 lag1 leftover 0.482. New groups cannot inherit H. |
| 3 | Who is turning? | **CLOSE unused leftover** leftover after days 0.453 vs days 0.711. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | Twin screen: none. Sister-runway footnote KEEP. |
| 6 | Months earlier? | lag1 leftover 0.482; days_lag1 0.684. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `h_sib_neg_share` as Y3 X / the 15-col card | **CLOSE unused leftover** | unused leftover after days: honest rank 0.453 dies (OLS 0.453 fake=False). Single 0.434 vs days 0.711 / size 0.617. DROP from the 44 as Y3 X. PARK as Y (H marks sister existence; hidden test is new groups). Q5 sister-runway footnote KEEP. |
| `h_sib_neg_share` as engine X on the 44 | **DROP** | leftover lives=False twin=False SIZE=False beat-size=False |
| H as health Y | **PARK** | sister existence; hidden test is new groups |
| Q5 sister mean b_runway ≥ 1 | **KEEP** | leftover after flag 0.425; raw gap 7.9% |
| mixed dummy / h_group_size twin | **NO** | ρ mix 0.025 size 0.089 |
| h_share_group_in | **NEAR_SIZE / not this stem** | Y3 0.638; ρ vs log1p(a_op_in) 0.796 |
| Q6 lag1 after days_lag1 | **CLOSE** | leftover 0.482 |

## 1 — Coverage; twin / SIZE screen

Train 1,214 co / 21,157 CM. h_sib_neg_share cov 94.9% acf1=0.048. Dark 470 = mixed 110 + all-dark 360 CONFIRM 110/360/470. ρ vs days 0.063 vs a_n_tx 0.053 vs h_group_size 0.089 vs mixed 0.025 vs size 0.046 vs own B -0.132. SIZE=False twins=none twin_gate=False.

| col | n_nn | cov | acf1 |
| --- | --- | --- | --- |
| h_sib_neg_share | 20,079 | 94.9% | 0.048 |
| h_group_size | 21,157 | 100.0% | — |
| h_share_group_in | 20,729 | 98.0% | 0.011 |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | 0.063 | no |
| a_n_tx | 0.053 | no |
| h_group_size | 0.089 | no |
| mixed_dummy | 0.025 | no |
| log1p(a_in3) | 0.046 | no |
| h_n_siblings_active | 0.145 | no |
| h_share_group_in | 0.003 | no |
| b_runway | -0.132 | no |


## 2 — Single-feature group-fold Y3

Y3 h_sib_neg_share 0.434 n=5,408 pos=366 (quote 0.434 CONFIRM). mixed 0.525 (quote 0.494) share_group_in 0.638 (quote 0.638 CONFIRM) vs size 0.617 vs days 0.711. Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=-0.183 FAIL.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | h_sib_neg_share | 5,408 | 366 | 0.434 | 0.504 | 0.066 | -1 | 0.332 0.446 0.419 0.514 0.457 |
| y3_recover_cash_6m | mixed_dummy | 5,648 | 402 | 0.525 | 0.531 | 0.019 | 1 | 0.507 0.509 0.532 0.522 0.554 |
| y3_recover_cash_6m | h_group_size | 5,648 | 402 | 0.552 | 0.580 | 0.119 | -1 | 0.738 0.435 0.501 0.493 0.593 |
| y3_recover_cash_6m | h_share_group_in | 5,583 | 386 | 0.638 | 0.639 | 0.073 | -1 | 0.523 0.669 0.723 0.630 0.646 |
| y3_recover_cash_6m | h_n_siblings_active | 5,648 | 402 | 0.548 | 0.575 | 0.118 | -1 | 0.732 0.431 0.485 0.503 0.590 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |


## 3 — Honest leftover after days

h_sib_neg_share leftover after days OLS 0.453 rank 0.453 fake=False almost=False ρ(resid,days)=-0.011 R²=0.004 honest_dies=True n=5,408 pos=366. Inverse: days leftover after sib_neg OLS 0.723 rank 0.724 dies=False.

OLS folds: 0.343 0.571 0.402 0.509 0.440. Rank folds: 0.342 0.569 0.404 0.507 0.441.

## 4 — Twin / SIZE screen (in cut 1)

SIZE=False twin_gate=False twins=none.

## 5 — Leftover after size; T1 residual

sib_neg leftover after size rank 0.446 dies=True. after h_group_size 0.426 dies=True. after mixed 0.435 dies=True. after days+size 0.450 dies=True. T1 mixed−all-dark residual 16.7% (quote +16.7pp CONFIRM). T1 residual is the mixed-existence gap, not leftover of h_sib_neg (after size dies=True).

| tercile | n | n_pos | CV | leftover | dies | mixed rate | all-dark rate | residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 1114 | 161 | 0.458 | 0.453 | True | 25.0% | 8.3% | 16.7% |
| T2 | 1931 | 84 | 0.542 | 0.554 | False | 10.1% | 4.7% | 5.4% |
| T3 | 2251 | 112 | 0.381 | 0.386 | True | 6.0% | 4.4% | 1.6% |


## 6 — Mixed 110 vs all-dark 360 leftover

Y3 mixed 12.1% vs all-dark 5.2% (quote 12.05% / 5.20%). mixed leftover after days 0.420 dies=True. all-dark leftover 0.375 dies=True. H leftover is sister existence, not health — hidden test is new groups.

| slice | n | n_pos | CV | leftover | dies | Y3 rate |
| --- | --- | --- | --- | --- | --- | --- |
| mixed_110 | 473 | 57 | 0.433 | 0.420 | True | 12.1% |
| all_dark_360 | 1434 | 65 | 0.384 | 0.375 | True | 5.2% |
| erp | 3501 | 244 | 0.502 | 0.539 | True | 7.3% |
| dark_470 | 1907 | 122 | 0.393 | 0.383 | True | 6.8% |


## 7 — Sister runway ≥ 1 footnote

110 sister runway≥1 Y3 15.6% n=257 vs stressed 7.7% n=208 gap 7.9% (quote +7.87pp raw / +6.97pp T1 CONFIRM). h_sib_neg leftover after runway flag 0.425 dies=True. H leftover after days on 110 0.412 dies=True. H median-split gap 2.2% (quote +0.01pp CONFIRM flat). Sister-runway flag Y3 0.602 (report-only; Y3 never B). Q5 footnote **KEEP**.

## 8 — Q6 lag1 leftover after days_lag1

Y3 sib_neg_lag1 0.475 leftover after days_lag1 rank 0.482 dies=True. Days lag1 0.684 (quote 0.684 CONFIRM). Hidden test is new groups — H cannot transfer.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | h_sib_neg_share | 5,408 | 366 | 0.434 | 0.504 | 0.066 | -1 | 0.332 0.446 0.419 0.514 0.457 |
| y3_recover_cash_6m | h_sib_neg_share_lag1 | 5,408 | 366 | 0.475 | 0.510 | 0.098 | 1 | 0.334 0.585 0.529 0.504 0.423 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | 0.694 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |


## 9 — vs `h_share_group_in`

ρ(sib_neg, share_group_in)=0.003. share vs log1p(a_in3) 0.677 vs log1p(a_op_in) 0.796 NEAR_SIZE CONFIRM. Y3 share 0.638 (quote 0.638). sib_neg leftover after share 0.422 dies=True. share leftover after days 0.554 dies=False.

## 10 — Holdout coverage only

Holdout 72 co / 1073 CM nn=963 cov=89.7% p50=0.400 dark co=32 (no fit, no AUROC).

## Extras

### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.404 p50=0.452 p95=0.543 share<0.55=97.5% n=40.

### Permute within days quintile

Permuted-within-days leftover rank p50=0.511 p90=0.519 n=24.

### ICC

Company ICC=0.897 k=1150. Group ICC=0.980 k=171. TRAIT at company; H is a group-type dummy (hidden test = new groups).

### Y3 rate by sib_neg quintile

Y3 sib_neg Q1→Q5 ['6.2%', '8.2%', '5.7%', '7.5%', '6.0%'].

| q | Y3 rate | n | n_pos |
| --- | --- | --- | --- |
| 1 | 6.2% | 1090 | 68 |
| 2 | 8.2% | 1105 | 91 |
| 3 | 5.7% | 1068 | 61 |
| 4 | 7.5% | 1176 | 88 |
| 5 | 6.0% | 969 | 58 |


### y11-style T1 residual

y11 T1 residual mixed−all-dark 12.5% (quote +12.5pp CONFIRM).

### mixed dummy leftover / dark-only

mixed dummy full Y3 0.525 leftover after days 0.646 dies=True fake=True; after size 0.561. Dark-only Y3 0.566 (quote 0.566 CONFIRM) leftover 0.450 dies=True. Existence dummy can look useful on this panel and still fail on new groups.

### sister runway≥1 T1 residual

110 sister runway≥1 T1 residual 7.0% (quote +6.97pp CONFIRM). H leftover after runway+days on 110 0.415 dies=True.

| tercile | n ok | n st | Y3 ok | Y3 st | residual |
| --- | --- | --- | --- | --- | --- |
| T1 | 64 | 52 | 28.1% | 21.2% | 7.0% |
| T2 | 61 | 68 | 16.4% | 4.4% | 12.0% |
| T3 | 131 | 86 | 8.4% | 2.3% | 6.1% |


### share_group_in leftover after size

share leftover after size 0.587 dies=False; after days 0.554 dies=False. NEAR_SIZE — do not promote share_group_in as leftover of sib_neg.

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| a_uncat_share leftover | 0.573 DROP / single 0.542 |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put H on the 15-col card.

## Files written

- `analysis/evaluate/sib_neg_qa.py`
- `analysis/outputs/sib_neg_qa.md`
- `analysis/outputs/sib_neg_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_sib_neg.md` (end, if WRITE_WAVE)

Elapsed 30s. Failed: none.

