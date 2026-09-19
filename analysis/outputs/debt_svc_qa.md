# Unused leftover of contemporaneous `a_debt_service` after days as Y3 X

Generated `2026-09-19T07:43:46+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_debt_service`. Do not put `a_debt_service` on the 15-col card. Do not overwrite `ds_r_qa.*`, `fc_r_qa.*`, `op_out_qa.*`, `in3_qa.*`. Do not grow TURNOVER. `f_ds_r` leftover 0.528 DROP stays. `f_fc_r` leftover 0.449 DROP stays.

`a_debt_service` = -sum(amount | grp = debt_service) this month. Store twins: `f_debt_service` / `f_ds_r` / `a_fin_cost` / `f_fc_r`.

## Headline

CLOSE unused leftover leftover-after-days rank 0.484 (OLS 0.631, fake=False). Y3 a_debt_service 0.613 vs days 0.711 vs size 0.617 vs f_ds_r 0.620. SIZE=False twin_gate=True twins=['f_ds_r', 'f_debt_service']. Inverse days-after-a_debt_service 0.673. Leftover after f_ds_r 0.587. after days+size 0.487. Q6 lag1 leftover 0.522. Demean leftover 0.544. Card: **CLOSE unused leftover** / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — do not invent y_debt_service. Size bar 0.617 stays. |
| 2 | Who is improving? | leftover after days 0.484. Month debt service ≠ 45→65. |
| 3 | Who is turning? | **CLOSE unused leftover** vs days 0.711. |
| 4 | Dip vs fall? | leftover after f_ds_r 0.587 — ratio rewrite? False. |
| 5 | Why did it change? | twins=['f_ds_r', 'f_debt_service']; SIZE=False. |
| 6 | Months earlier? | lag1 leftover 0.522; days_lag1 0.684. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `a_debt_service` as Y3 X / 15-col card | **CLOSE unused leftover** | unused leftover after days: honest rank 0.484 dies (OLS 0.631 fake=False). DROP from the 44 as Y3 X. Off the 15-col card. Do not grow TURNOVER. |
| `a_debt_service` as engine X on the 44 | **DROP** | leftover lives=False twin=True SIZE=False beat-size=False |
| rewrite of f_ds_r | **NO** | leftover after f_ds_r 0.587 |
| `y_debt_service` | **PARK** | do not invent unless KEEP-as-X |
| Q6 lag1 / TURNOVER | **CLOSE** | leftover 0.522; do not grow 0.720 |
| `f_ds_r` leftover | **DROP quote 0.528** | store flow stays |
| `f_fc_r` leftover | **DROP quote 0.449** | KEEP f_fc_r_lag3 on TURNOVER |
| size bar `log1p(a_in3)` | **KEEP quote** | 0.617 stays |

## 1 — Coverage; twin / SIZE

Train 1,214 co / 21,157 CM. a_debt_service nn=21,157 cov 100.0% eq0 76.4% p50=0.000. Dark panel p50=0.000 ERP p50=0.000. max |a_debt_service - f_debt_service|=0.000. ρ vs days 0.382 vs a_n_tx 0.379 vs f_ds_r 0.881 vs f_debt_service 1.000 vs a_fin_cost 0.331 vs a_op_out 0.331 vs size 0.322 vs f_fc_r 0.194. SIZE=False gate_twins=['f_ds_r', 'f_debt_service'].

| col | n_nn | cov | eq0 | p50 |
| --- | --- | --- | --- | --- |
| a_debt_service | 21,157 | 100.0% | 76.4% | 0.000 |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | 0.382 | no |
| a_n_tx | 0.379 | no |
| f_ds_r | 0.881 | TWIN |
| f_debt_service | 1.000 | TWIN |
| a_fin_cost | 0.331 | no |
| a_op_out | 0.331 | no |
| log_in3 | 0.322 | no |
| f_fc_r | 0.194 | no |


## 2 — Single-feature group-fold Y3

Y3 a_debt_service 0.613 n=5,648 pos=402. vs size 0.617 vs days 0.711 vs f_ds_r 0.620 vs f_debt_service 0.613 vs a_fin_cost 0.634 vs f_fc_r 0.559 vs a_op_out 0.678. Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=-0.004 FAIL. f_ds_r leftover 0.528 DROP stays. Do not grow TURNOVER.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_debt_service | 5,648 | 402 | 0.613 | 0.618 | 0.034 | -1 | 0.570 0.643 0.650 0.591 0.613 |
| y3_recover_cash_6m | log1p(a_debt_service) | 5,648 | 402 | 0.613 | 0.618 | 0.034 | -1 | 0.570 0.643 0.650 0.591 0.613 |
| y3_recover_cash_6m | f_debt_service | 5,648 | 402 | 0.613 | 0.618 | 0.034 | -1 | 0.570 0.643 0.650 0.591 0.613 |
| y3_recover_cash_6m | f_ds_r | 5,528 | 391 | 0.620 | 0.625 | 0.045 | -1 | 0.547 0.633 0.671 0.618 0.627 |
| y3_recover_cash_6m | a_fin_cost | 5,648 | 402 | 0.634 | 0.626 | 0.049 | -1 | 0.607 0.666 0.703 0.583 0.613 |
| y3_recover_cash_6m | f_fc_r | 5,528 | 391 | 0.559 | 0.541 | 0.084 | -1 | 0.550 0.541 0.703 0.500 0.500 |
| y3_recover_cash_6m | a_op_out | 5,648 | 402 | 0.678 | 0.681 | 0.032 | -1 | 0.659 0.639 0.722 0.675 0.694 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |


## 3 — Honest leftover after days

a_debt_service leftover after days OLS 0.631 rank 0.484 fake=False almost=True ρ(resid,days)=-0.742 R²=0.004 honest_dies=True n=5,648 pos=402. Inverse: days leftover after a_debt_service OLS 0.709 rank 0.673 dies=False.

OLS folds: 0.588 0.627 0.606 0.642 0.691. Rank folds: 0.496 0.531 0.435 0.464 0.493.

## 4 — Leftover after f_ds_r

leftover after f_ds_r rank 0.587 OLS 0.538 dies=False fake=False rewrite=False. after days+f_ds_r 0.477. f_ds_r leftover after a_debt_service 0.545. f_ds_r leftover after days 0.528 DROP stays.

| cut | rank | ols | fake |
| --- | --- | --- | --- |
| after f_ds_r | 0.587 | 0.538 | False |
| after days+f_ds_r | 0.477 | 0.662 | False |
| f_ds_r after a_debt_service | 0.545 | 0.613 | True |


## 5 — Leftover after size / days+size

leftover after size 0.556 dies=False fake=False ρ=-0.675. after days+size 0.487 dies=True fake=False. Size bar 0.617 stays.

## 6 — Q6 lag1 leftover after days_lag1

Y3 a_debt_service_lag1 0.611 leftover after days_lag1 rank 0.522 dies=True. Days lag1 0.684 (quote 0.684 CONFIRM).

| feat | Y3 | leftover |
| --- | --- | --- |
| a_debt_service_lag1 | 0.611 | 0.522 |
| days_lag1 | 0.684 | — |


## 7 — Dark vs ERP

Dark 470 (want 470) last-month p50=0.000 ERP last p50=0.000. Dark leftover 0.474 ERP leftover 0.454.

| slice | p50 | leftover | n |
| --- | --- | --- | --- |
| Dark last-month | 0.000 | 0.474 | 2030 |
| ERP last-month | 0.000 | 0.454 | 3618 |


## 8 — Holdout coverage only

Holdout 72 co / 1073 CM nn=1073 cov=100.0% p50=0.000 (no fit, no AUROC).

## Extras

### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.447 p50=0.514 p95=0.548 share<0.55=95.0% n=40.

### ICC / acf1

ICC=0.755 median company Pearson acf1=0.026 n_co=483. month shock / WITHIN.

### leftover after twins / fin_cost / op_out

leftover after f_debt_service 0.613 a_fin_cost 0.559 a_op_out 0.519 a_n_tx 0.485 f_fc_r 0.595.

| control | leftover | ols | fake | dies |
| --- | --- | --- | --- | --- |
| f_debt_service | 0.613 | 0.613 | True | True |
| a_fin_cost | 0.559 | 0.558 | False | False |
| a_op_out | 0.519 | 0.518 | False | True |
| a_n_tx | 0.485 | 0.603 | False | True |
| f_fc_r | 0.595 | 0.615 | False | False |


### leftover of company-demeaned a_debt_service

demeaned a_debt_service Y3 0.449 leftover after days 0.544 dies=True fake=False.

### zero dummy / log1p leftover

zero-ds dummy Y3 0.614 leftover after days 0.473 fake=False. log1p(a_debt_service) Y3 0.613 leftover 0.484 dies=True.

### Y7 / Y2 leftover after days

Y7 a_debt_service 0.523 leftover after days 0.529 dies=True. Do not grow TURNOVER 0.720. Y2 0.526 leftover 0.465 dies=True.

### leftover on ds>0 / size terciles

ds>0 leftover 0.640 T1 0.526 T2 0.466 T3 0.610.

| slice | Y3 | leftover | fake | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| ds>0 | 0.499 | 0.640 | False | 1952 | 51 |
| T1_small | 0.540 | 0.526 | False | 1331 | 222 |
| T2 | 0.550 | 0.466 | False | 2084 | 116 |
| T3_large | 0.635 | 0.610 | False | 2233 | 64 |


### leftover of intensity / stacked controls

a_debt_service/a_in3 Y3 0.611 leftover after days 0.517. after days+size+f_ds_r 0.487 fake=False. after days+f_debt_service 0.574.

### leftover on ds>0 after days+f_ds_r / size

ds>0 Y3 0.499 leftover after days+f_ds_r 0.552 after days+size 0.613 after f_ds_r 0.499 ρ vs f_ds_r 0.544 n=1924.

### leftover on ever-ds / year

ever-ds leftover 0.534 n=2999. never-ds leftover 0.667. 2025 leftover 0.489 2026 leftover 0.450.

### leftover of f_ds_r / f_debt_service / f_fc_r after days

f_ds_r leftover after days 0.528 (quote 0.528 CONFIRM). f_debt_service leftover 0.484. f_fc_r leftover 0.449 (quote 0.449).

### last-labeled / MoM / leftover after days+a_fin_cost

last-labeled Y3 0.601 leftover 0.453 n=725. MoM Δ Y3 0.508 leftover 0.600 dies=False. after days+a_fin_cost 0.513. T3 leftover after days+f_ds_r 0.591.

### leftover of has-ds dummy / a_debt_service/a_op_out / last-3

has-ds dummy Y3 0.614 leftover after days 0.473 after days+size 0.476. a_debt_service/a_op_out Y3 0.607 leftover 0.519. last-3 leftover 0.472 n=1901.

### MoM leftover after days+f_ds_r; T3 leftover after days+size+f_ds_r

MoM leftover after days 0.600 fake=False ρ(resid,days)=-0.762. after days+f_ds_r 0.556 after days+size 0.528. T3 leftover after days 0.610 after days+size+f_ds_r 0.607 n=2187.

### leftover of 3m rolling / ever-ds share / last-labeled ever-ds

3m rolling Y3 0.618 leftover 0.527 dies=True. ever-ds share Y3 0.613 leftover 0.524. last-labeled ever-ds Y3 — leftover — n=333.

### T3 leftover after days+size / a_fin_cost; never-ds leftover fake?

T3 Y3 0.635 leftover after days+size 0.619 after days+a_fin_cost 0.609 after days+size+f_ds_r+a_fin_cost 0.600 fake=False n=2187. never-ds leftover 0.667 fake=False ρ=-0.742.

### leftover of a_debt_service/a_n_tx; leftover after days+f_fc_r

a_debt_service/a_n_tx Y3 0.609 leftover 0.482. after days+f_fc_r 0.512. after days+a_n_tx+f_ds_r 0.480. Dark last-labeled leftover — n=0.

### T3 leftover after days+a_n_tx / a_op_out; T3 ds>0 leftover

T3 leftover after days+a_n_tx 0.615 after days+a_op_out 0.599 after days+size+f_ds_r+a_fin_cost+a_n_tx 0.599 fake=False. T3 ds>0 Y3 — leftover — n=1146. T3 last-labeled leftover — n=252.

### leftover of within-company rank / leftover after days+f_ds_r+a_n_tx

within-co rank Y3 0.462 leftover 0.530 dies=True. after days+f_ds_r+a_n_tx 0.480. high-days leftover 0.604 low-days leftover 0.555.

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| f_ds_r leftover | 0.528 DROP |
| f_fc_r leftover | 0.449 DROP |
| q6_keep | issued_lag1 / days_lag1 / ss_lag1 |

Do not grow TURNOVER. Do not put `a_debt_service` on the 15-col card. KEEP `f_fc_r_lag3` on TURNOVER. Do not quote a_out_vol 0.722 as the engine.

## Files written

- `analysis/evaluate/debt_svc_qa.py`
- `analysis/outputs/debt_svc_qa.md`
- `analysis/outputs/debt_svc_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_debt_svc.md` (end, if WRITE_WAVE)

Elapsed 14s. Failed: none.

