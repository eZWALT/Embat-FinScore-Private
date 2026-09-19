# Unused leftover of `a_op_out` after days as Y3 X

Generated `2026-09-19T07:37:44+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_op_out`. Do not put a_op_out on the 15-col card. Do not overwrite `a_vol_qa.*`, `transfer_qa.*`, `growth_qa.*`, `in3_qa.*`, `n_accounts_qa.*`. Do not quote a_out_vol 0.722 as the engine. Do not grow TURNOVER.

`a_op_out` = -sum(amount | grp = op_out) this month. Trailing twins: `a_out3` / `a_out6` / `a_out12`. `a_in3` leftover 0.521 already CLOSE. Size bar 0.617 stays.

## Headline

CLOSE unused leftover leftover-after-days rank 0.586 (OLS 0.675, fake=True). Y3 a_op_out 0.678 vs days 0.711 vs size 0.617 vs a_out3 0.601. SIZE=True twin_gate=True twins=['a_out3', 'a_out6']. Inverse days-after-a_op_out 0.638. Leftover after a_out3 0.617. after days+size 0.576. Q6 lag1 leftover 0.561. Demean leftover 0.525 dies. mid-quintile leftover 0.469 dies; T3 leftover 0.438 dies. leftover after days+CV 0.518 dies (vol ate leftover). Card: **CLOSE unused leftover** / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — do not invent y_op_out. Size bar 0.617 stays. |
| 2 | Who is improving? | leftover after days 0.586. Month outflow ≠ 45→65. |
| 3 | Who is turning? | **CLOSE unused leftover** vs days 0.711. |
| 4 | Dip vs fall? | leftover after a_out3 0.617 — trailing rewrite? False. |
| 5 | Why did it change? | twins=['a_out3', 'a_out6']; SIZE=True. |
| 6 | Months earlier? | lag1 leftover 0.561; days_lag1 0.684. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `a_op_out` as Y3 X / 15-col card | **CLOSE unused leftover** | honest leftover after days rank 0.586 lives (OLS 0.675 fake=True ρ(resid,days)=-0.841). TWIN of ['a_out3', 'a_out6']. SIZE (|ρ| vs log1p(a_in3) ≥0.50). DROP from the 44. Off the 15-col card. |
| `a_op_out` as engine X on the 44 | **DROP** | leftover lives=True twin=True SIZE=True beat-size=True |
| rewrite of a_out3 | **NO** | leftover after a_out3 0.617 |
| `y_op_out` | **PARK** | do not invent unless KEEP-as-X |
| Q6 lag1 / TURNOVER | **CLOSE** | leftover 0.561; do not grow 0.720 |
| size bar `log1p(a_in3)` | **KEEP quote** | 0.617 stays; a_in3 leftover 0.521 CLOSE |

## 1 — Coverage; twin / SIZE

Train 1,214 co / 21,157 CM. a_op_out nn=21,157 cov 100.0% eq0 12.7% p50=68794.250. Dark panel p50=80196.140 ERP p50=64746.835. ρ vs days 0.609 vs a_n_tx 0.655 vs a_out3 0.907 vs a_out6 0.860 vs a_out12 0.799 vs a_net -0.269 vs a_io_ratio -0.070 vs size 0.741. SIZE=True gate_twins=['a_out3', 'a_out6'].

| col | n_nn | cov | eq0 | p50 |
| --- | --- | --- | --- | --- |
| a_op_out | 21,157 | 100.0% | 12.7% | 68794.250 |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | 0.609 | no |
| a_n_tx | 0.655 | no |
| a_out3 | 0.907 | TWIN |
| a_out6 | 0.860 | TWIN |
| a_out12 | 0.799 | no |
| a_net | -0.269 | no |
| a_io_ratio | -0.070 | no |
| log_in3 | 0.741 | SIZE |
| a_op_in | 0.736 | no |


## 2 — Single-feature group-fold Y3

Y3 a_op_out 0.678 n=5,648 pos=402. vs size 0.617 vs days 0.711 vs a_out3 0.601 vs a_out6 0.607 vs a_out12 0.575 vs a_net 0.475 vs a_io_ratio 0.565. Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=0.061 PASS. Do not quote a_out_vol 0.722 as the engine.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_op_out | 5,648 | 402 | 0.678 | 0.681 | 0.032 | -1 | 0.659 0.639 0.722 0.675 0.694 |
| y3_recover_cash_6m | log1p(a_op_out) | 5,648 | 402 | 0.678 | 0.681 | 0.032 | -1 | 0.658 0.639 0.722 0.676 0.694 |
| y3_recover_cash_6m | a_out3 | 5,528 | 391 | 0.601 | 0.597 | 0.049 | -1 | 0.582 0.568 0.673 0.555 0.628 |
| y3_recover_cash_6m | log1p(a_out3) | 5,528 | 391 | 0.601 | 0.597 | 0.049 | -1 | 0.582 0.568 0.673 0.555 0.628 |
| y3_recover_cash_6m | a_out6 | 4,212 | 313 | 0.607 | 0.603 | 0.060 | -1 | 0.573 0.547 0.686 0.574 0.654 |
| y3_recover_cash_6m | a_out12 | 1,925 | 150 | 0.575 | 0.579 | 0.049 | -1 | 0.575 0.508 0.557 0.595 0.641 |
| y3_recover_cash_6m | a_net | 5,648 | 402 | 0.475 | 0.516 | 0.056 | 1 | 0.402 0.435 0.485 0.537 0.517 |
| y3_recover_cash_6m | a_io_ratio | 5,528 | 391 | 0.565 | 0.563 | 0.043 | -1 | 0.523 0.578 0.632 0.550 0.541 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |


## 3 — Honest leftover after days

a_op_out leftover after days OLS 0.675 rank 0.586 fake=True almost=True ρ(resid,days)=-0.841 R²=0.001 honest_dies=True n=5,648 pos=402. Inverse: days leftover after a_op_out OLS 0.709 rank 0.638 dies=False.

OLS folds: 0.630 0.691 0.684 0.678 0.693. Rank folds: 0.605 0.504 0.660 0.599 0.563.

## 4 — Leftover after a_out3

leftover after a_out3 rank 0.617 OLS 0.545 dies=False fake=False rewrite=False. after days+a_out3 0.572. a_out3 leftover after a_op_out 0.543.

| cut | rank | ols | fake |
| --- | --- | --- | --- |
| after a_out3 | 0.617 | 0.545 | False |
| after days+a_out3 | 0.572 | 0.586 | False |
| a_out3 after a_op_out | 0.543 | 0.525 | False |


## 5 — Leftover after size / days+size

leftover after size 0.623 dies=True fake=True ρ=-0.866. after days+size 0.576 dies=False fake=False. a_in3 leftover 0.521 CLOSE stays. Size bar 0.617 stays.

## 6 — Q6 lag1 leftover after days_lag1

Y3 a_op_out_lag1 0.640 leftover after days_lag1 rank 0.561 dies=True. Days lag1 0.684 (quote 0.684 CONFIRM).

| feat | Y3 | leftover |
| --- | --- | --- |
| a_op_out_lag1 | 0.640 | 0.561 |
| days_lag1 | 0.684 | — |


## 7 — Dark vs ERP

Dark 470 (want 470) last-month p50=56139.765 ERP last p50=50646.965. Dark leftover 0.632 ERP leftover 0.581.

| slice | p50 | leftover | n |
| --- | --- | --- | --- |
| Dark last-month | 56139.765 | 0.632 | 2030 |
| ERP last-month | 50646.965 | 0.581 | 3618 |


## 8 — Holdout coverage only

Holdout 72 co / 1073 CM nn=1073 cov=100.0% p50=54289.740 (no fit, no AUROC).

## Extras

### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.568 p50=0.590 p95=0.644 share<0.55=0.0% n=40.

### ICC / acf1

ICC=0.955 median company Pearson acf1=0.008 n_co=1199. BETWEEN trait.

### leftover after trailing / net / io / n_tx

leftover after a_out6 0.623 a_out12 0.622 a_net 0.685 a_io_ratio 0.680 a_n_tx 0.576.

| control | leftover | ols | fake | dies |
| --- | --- | --- | --- | --- |
| a_out6 | 0.623 | 0.538 | False | False |
| a_out12 | 0.622 | 0.479 | False | False |
| a_net | 0.685 | 0.658 | False | False |
| a_io_ratio | 0.680 | 0.679 | False | False |
| a_n_tx | 0.576 | 0.652 | True | True |


### zero dummy / log1p leftover

zero-out dummy Y3 0.560 leftover after days 0.596 fake=False. log1p(a_op_out) Y3 0.678 leftover 0.586 dies=False.

### Y7 leftover after days

Y7 a_op_out 0.537 leftover after days 0.535 dies=True. Do not grow TURNOVER 0.720.

### Y2 leftover after days

Y2 a_op_out 0.485 leftover after days 0.450 dies=True.

### leftover after a_op_in

leftover after a_op_in 0.598. after days+a_op_in 0.564. a_op_in Y3 0.676 leftover after days 0.584.

### leftover of a_out3 after days / days+n_tx / days+size+out3

a_out3 Y3 0.601 leftover after days 0.517 dies=True fake=True. a_op_out leftover after days+a_n_tx 0.577 fake=False. after days+size+a_out3 0.572 dies=False.

### leftover of company-demeaned a_op_out

demeaned a_op_out Y3 0.485 leftover after days 0.525 dies=True fake=False. ICC 0.955 BETWEEN + acf1 0.008 — demean kills the trait.

### leftover on out>0 / size terciles

out>0 leftover 0.541 T1 0.575 T2 0.553 T3 0.438.

| slice | Y3 | leftover | fake | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| out>0 | 0.647 | 0.541 | True | 5332 | 332 |
| T1_small | 0.622 | 0.575 | True | 1331 | 222 |
| T2 | 0.477 | 0.553 | True | 2084 | 116 |
| T3_large | 0.536 | 0.438 | True | 2233 | 64 |


### leftover of a_op_out / a_out3

a_op_out/a_out3 Y3 0.608 leftover after days 0.583 dies=False fake=False. after days+a_out3 0.583.

### leftover after size / a_n_tx fake flags

leftover after size rank 0.623 OLS 0.598 fake=True ρ=-0.866. leftover after a_n_tx rank 0.576 fake=True ρ=-0.818. leftover after days+a_out6 0.563 fake=False.

### leftover of month shock a_op_out - a_out3/3

month-shock (a_op_out-a_out3/3) Y3 0.561 leftover after days 0.548 dies=True fake=False. after days+size 0.545.

### last-month / first-half / last-half / trail length

last-month Y3 — leftover — n=0. first-half leftover 0.585 last-half 0.566. trail≥12 leftover 0.577 short —.

### MoM Δ leftover / a_op_out/a_op_in

MoM Δ a_op_out Y3 0.536 leftover after days 0.524 dies=True fake=False. a_op_out/a_op_in Y3 0.550 leftover 0.533 dies=True.

### leftover by year / stacked controls

2025 leftover 0.561 n=4208. 2026 leftover 0.635 n=955. a_out3-defined leftover 0.582. after days+size+out3+n_tx 0.575 fake=False. after a_out3+size 0.614.

### leftover on sibling groups / never-zero

sibling n_co≥2 leftover 0.588 n=5408. solo leftover —. never-zero leftover 0.535 ever-zero leftover 0.538.

### last / first labeled leftover

last-labeled Y3 0.729 leftover 0.660 n=725 pos=154. first-labeled leftover 0.580 n=725. days>0 leftover 0.569.

### winsor / mid-quintile / log after size

winsor 1/99 Y3 0.678 leftover 0.586. mid-quintile leftover 0.469 tail leftover 0.611. log_opout leftover after size 0.623 fake=False. leftover after days+io 0.585.

### last-labeled after size / outflow per day

last-labeled leftover after size 0.704 fake=True. after days+size 0.672. outflow/day Y3 0.630 leftover after days 0.569 after days+size 0.563. Q5 leftover 0.695 Q1 leftover 0.456.

### leftover after full stack / high vs low median out

leftover after days+size+out3+n_tx+io 0.575 fake=False. after days+a_out12 0.559. log leftover after days+size 0.575. high-median-out leftover 0.622 low-median-out leftover 0.454.

### last-labeled Dark vs ERP

last-labeled Dark leftover 0.683 n=262 ERP leftover 0.659 n=463. Dark after size 0.693 ERP after size 0.718. last-labeled leftover after a_out3 0.672.

### leftover by onboard / company ρ(out, size)

early-onboard leftover 0.601 late-onboard leftover 0.598. high ρ(out,size) leftover 0.482 low ρ leftover 0.695 median company ρ=0.336 n_co=347.

### leftover after days+size / out3 on low vs high ρ

low-ρ Y3 0.767 leftover after days+size 0.603 after days+out3 0.581 a_out3 leftover 0.678 ρ vs a_out3 0.830 n=2155. high-ρ Y3 0.596 leftover after days+size 0.581 after days+out3 0.630 ρ vs a_out3 0.910 n=2205.

### leftover of a_op_out/a_in3 / net-out months

a_op_out/a_in3 Y3 0.612 leftover after days 0.583 after days+size 0.579. net-out leftover 0.525 net-in leftover 0.662. high-n_tx leftover 0.590 low-n_tx leftover 0.574.

### leftover on ever-Y3 / io slice / last-3

ever-Y3 leftover 0.566 n=919 never-Y3 leftover —. io≥1 leftover 0.609 io<1 leftover 0.569. last-3-labeled leftover 0.604 n=1901. log_opout-log_in3 Y3 0.548 leftover 0.551.

### leftover after days+size on last-3 / 2026

last-3 leftover after days+size 0.601 after days+out3 0.580. last-3 2026 leftover 0.635 n=955. last-labeled leftover after days+size+out3 0.654. high a_n_tx leftover 0.562.

### leftover on low company ρ(out, out3)

company ρ(out,out3)<0.80 leftover 0.614 Y3 0.703 n=3896 n_co=304 panel ρ vs a_out3 0.854. after days+size 0.608. twin-co leftover — n_co=47 median company ρ=0.555.

### leftover on a_out3-missing / young companies

a_out3-missing leftover — Y3 — n=120. age≤3 leftover — after days+size —. age>12 leftover 0.569.

### leftover on mature trails / Dark 2026

age 4–12 leftover 0.573 n=3504. age>12 leftover after days+size 0.580 after days+out3 0.573. Dark 2026 leftover — n=356 ERP 2026 leftover 0.629 n=599.

### leftover on stable-outflow / size IQR

stable-outflow leftover 0.437 noisy leftover 0.565. size-IQR leftover 0.562. age>12 leftover after days+size+out3 0.573.

### leftover after days+size on noisy / leftover of CV

noisy leftover after days+size 0.557 after days+out3 0.582. outflow CV Y3 0.795 leftover after days 0.734 dies=False.

### leftover of outflow CV after size (PARK)

outflow CV Y3 0.795 leftover after size 0.763 fake=False ρ vs size -0.367. after days+size 0.732. ρ vs days -0.469. PARK — do not put CV on the 15-col card; this is not a_op_out KEEP.

### leftover of outflow CV after n_tx / out3 (PARK)

CV leftover after a_n_tx 0.735 ρ=-0.478. after a_out3 0.760 ρ=-0.448. after days+n_tx+size 0.732. PARK — not a_op_out KEEP; off the card.

### leftover of a_op_out after days+CV

a_op_out leftover after days+CV 0.518 fake=False. after days+CV+size 0.469. after days+CV+out3 0.534.

### leftover of CV after a_op_out (inverse)

CV leftover after a_op_out 0.730 fake=False. after a_op_out+days 0.717. a_op_out leftover after days+CV died 0.518 — vol ate the leftover.

### leftover of a_op_out after CV-only / last-labeled days+CV

a_op_out leftover after CV-only 0.548 fake=False. last-labeled leftover after days+CV 0.586 n=719.

### leftover after days+CV on Dark / ERP / T3

leftover after days+CV Dark 0.562 ERP 0.440 T3 0.469 2026 0.582.

### leftover after days+CV on Q5 / mid / net-out

leftover after days+CV Q5 0.766 mid 0.638 net-out 0.549.

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| a_in3 leftover | 0.521 CLOSE |
| q6_keep | issued_lag1 / days_lag1 / ss_lag1 |

Do not quote a_out_vol 0.722 as the engine. Do not grow TURNOVER. Do not put `a_op_out` on the 15-col card.

## Files written

- `analysis/evaluate/op_out_qa.py`
- `analysis/outputs/op_out_qa.md`
- `analysis/outputs/op_out_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_op_out.md` (end, if WRITE_WAVE)

Elapsed 18s. Failed: none.

