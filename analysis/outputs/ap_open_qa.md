# Unused leftover of `e_ap_open` after `c_n_days_with_tx`

Generated `2026-09-19T06:47:37+02:00` by agent `e8b2c0d4`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_ap_open`. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Do **not** grow TURNOVER. Do **not** put e_ap_open on the 15-col Y3 card. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. AP issued leftover 0.591 CLOSED — do not overwrite ap_issued_qa. DPO already DROP — do not overwrite dpo_qa.

`e_ap_open` = unpaid AP |amount| stock at period end. Feature report: 64.1% cov, acf1 0.74, ICC 0.99 BETWEEN. `e_dpo_proxy` = open / this-period issued.

## Headline

CLOSE leftover-after-days rank 0.418 (OLS 0.695, fake=False). Y3 open 0.569 vs days 0.711 vs size 0.617 vs issued 0.675 vs DPO 0.627. SIZE=False twin=False. Inverse days-after-open 0.720. Leftover after issued 0.459. Leftover after DPO 0.605. Card: CLOSE unused leftover / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| open leftover after days (Y3 X) | **CLOSE** | Y3 leftover after days rank 0.418 OLS 0.695 dies; twin=False SIZE=False. Inverse days-after-open 0.720. Contemporaneous AP open stays off the 15-col card. Do not grow TURNOVER. Open leftover after issued dies — rewrite of issued volume. |
| 15-col Y3 card stem | **KEEP off the card** | do not put e_ap_open on the card |
| TURNOVER add-on | **CLOSE** | do not grow 0.720 |
| AP issued leftover | **CLOSE (locked)** | rank 0.591 fake days clone |
| DPO | **DROP (locked)** | do not reopen dpo_qa |
| health Y `y_ap_open` | **PARK** | do not invent y_ap_open |

## 1. Coverage / 470 vs ERP / SIZE ρ

Train e_ap_open nn=13,554 cov=64.1% (feature report 64.1%). Dark never-ERP 470 (want 470): nn=0 zero=0 pos=0 CONFIRM NaN not 0. Ever-ERP 744 nn=13,554 of which zero=1,381. p50=49042 p99=66154829 max=5256380469. ρ vs log1p(a_in3)=0.462 n=12066 not SIZE (quote 0.411). acf1=0.744 (quote 0.74) ICC=0.988 (quote 0.99 BETWEEN) w/b=0.013 k=744.

| slice | n_cm | nn | cov | eq0 | p50 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| all train | 21,157 | 13,554 | 64.1% | 10.2% | 49042 | 66154829 |
| ever-ERP | 13,554 | 13,554 | 100.0% | 10.2% | — | — |
| dark 470 | 7,603 | 0 | 0.0% | — | — | — |


## 2. Spearman twins / SIZE

ap_open vs days ρ=0.379 (not a twin). vs a_n_tx 0.423 vs e_ap_issued 0.757 vs DPO 0.460 vs pending 0.250 vs log1p(a_in3) 0.462 not SIZE. vs e_ar_open 0.682 vs open_lag1 0.944. TWIN |ρ|≥0.80: e_ap_open_lag1. Gate twins: none.

| vs | ρ | n | twin? |
| --- | --- | --- | --- |
| c_n_days_with_tx | 0.379 | 13,554 |  |
| a_n_tx | 0.423 | 13,554 |  |
| e_ap_issued | 0.757 | 13,554 |  |
| e_dpo_proxy | 0.460 | 10,829 |  |
| e_pending_amt_share | 0.250 | 12,762 |  |
| log1p(a_in3) | 0.462 | 12,066 |  |
| e_ar_open | 0.682 | 13,554 |  |
| e_ap_open_lag1 | 0.944 | 12,810 | TWIN |


## 3. Single-feature group-fold Y3 / Y7

Y3 e_ap_open 0.569 vs days 0.711 (CONFIRM 0.711) vs size 0.617 (CONFIRM 0.617) vs e_ap_issued 0.675 (CONFIRM 0.675) vs DPO 0.627 vs e_ar_open 0.587. open_lag1 0.561. Y7 open 0.460. Beat size -0.047.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | e_ap_open | 3,618 | 264 | 0.569 | 0.109 | -1 | 0.638 0.674 0.461 0.443 0.632 |
| y3_recover_cash_6m | e_ap_open_lag1 | 3,618 | 264 | 0.561 | 0.102 | -1 | 0.625 0.668 0.479 0.430 0.602 |
| y3_recover_cash_6m | log1p_open | 3,618 | 264 | 0.569 | 0.109 | -1 | 0.638 0.674 0.461 0.443 0.632 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | e_ap_issued | 3,618 | 264 | 0.675 | 0.117 | -1 | 0.811 0.656 0.492 0.712 0.704 |
| y3_recover_cash_6m | e_dpo_proxy | 3,015 | 174 | 0.627 | 0.112 | 1 | 0.797 0.510 0.605 0.669 0.554 |
| y3_recover_cash_6m | e_ar_open | 3,618 | 264 | 0.587 | 0.101 | -1 | 0.544 0.738 0.547 0.476 0.629 |
| y3_recover_cash_6m | e_pending_amt_share | 3,446 | 239 | 0.484 | 0.102 | 1 | 0.571 0.458 0.381 0.609 0.399 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y7_top1_lost | e_ap_open | 7,464 | 2,149 | 0.460 | 0.081 | -1 | 0.527 0.410 0.530 0.488 0.343 |
| y7_top1_lost | e_ap_open_lag1 | 7,253 | 2,072 | 0.465 | 0.074 | -1 | 0.528 0.420 0.529 0.489 0.360 |
| y7_top1_lost | log1p_open | 7,464 | 2,149 | 0.460 | 0.081 | -1 | 0.527 0.410 0.530 0.488 0.343 |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 |
| y7_top1_lost | log1p_a_in3 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 |
| y7_top1_lost | e_ap_issued | 7,464 | 2,149 | 0.570 | 0.058 | -1 | 0.562 0.483 0.572 0.589 0.644 |
| y7_top1_lost | e_dpo_proxy | 7,038 | 1,929 | 0.450 | 0.051 | -1 | 0.497 0.403 0.510 0.438 0.403 |
| y7_top1_lost | e_ar_open | 7,464 | 2,149 | 0.465 | 0.092 | -1 | 0.508 0.514 0.493 0.511 0.301 |
| y7_top1_lost | e_pending_amt_share | 7,464 | 2,149 | 0.420 | 0.040 | -1 | 0.410 0.410 0.467 0.450 0.363 |
| y7_top1_lost | a_n_tx | 7,464 | 2,149 | 0.452 | 0.032 | -1 | 0.461 0.482 0.476 0.436 0.405 |


## 4. Honest leftover after days (Y3)

Y3 leftover after days rank 0.418 OLS 0.695 ρ(resid,days)=-0.672 R²=0.000 dies. Inverse: days leftover after open rank 0.720 survives — keep the 0.711 bar.

| bar | rank | OLS | ρ(resid,ctrl) | R² | fake? | n |
| --- | --- | --- | --- | --- | --- | --- |
| open leftover after days | 0.418 | 0.695 | -0.672 | 0.000 |  | 3,618 |
| days leftover after open | 0.720 | 0.731 | 0.348 | 0.000 |  | 3,618 |


## 5. Leftover after e_ap_issued

Y3 leftover after e_ap_issued rank 0.459 OLS 0.560 ρ=0.712 REWRITE of issued volume. After issued+days 0.549 fake=False.

| bar | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| open leftover after issued | 0.459 | 0.560 | 0.712 |  | 3,618 |
| open leftover after issued+days | 0.549 | 0.691 | 0.038 |  | 3,618 |


## 6. Leftover after DPO

Y3 leftover after DPO rank 0.605 OLS 0.485 ρ=0.476 not just the DPO numerator. After DPO+days 0.429. DPO already DROP — do not reopen.

| bar | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| open leftover after DPO | 0.605 | 0.485 | 0.476 |  | 3,015 |
| open leftover after DPO+days | 0.429 | 0.676 | 0.179 |  | 3,015 |


## 7. Dark 470

Dark never-ERP 470 nn=0 zero=0 pos=0. CONFIRM NaN not 0.

## 8. Q6 lag1 leftover after days_lag1

Q6 Y7 open_lag1 on short 0.478 (AR issued_lag1 KEEP 0.626 locked). open_lag1 leftover after days_lag1 short rank 0.442 fake=False. Contemporaneous leftover after days_lag1 short 0.517 fake=False. Do not claim a TURNOVER seat.

| slice | CV | n | n_pos |
| --- | --- | --- | --- |
| Q6 Y7 open_lag1 short | 0.478 | 4,210 | 1,260 |
| Q6 Y3 open_lag1 leftover after days_lag1 short |  |  |  |
| Q6 Y3 now leftover after days_lag1 short |  |  |  |


## 9. vs e_ar_open

AP leftover after AR open 0.423 same object. AR leftover after AP 0.573. After AR+days 0.477.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| AP open leftover after AR open | 0.423 | 0.546 | 0.081 |  |
| AR open leftover after AP open | 0.573 | 0.542 | -0.581 |  |
| AP open leftover after AR+days | 0.477 | 0.517 | 0.068 |  |


## 10. Bootstrap leftover-after-days

Bootstrap leftover-after-days rank p05=0.399 p50=0.455 p95=0.571 n=40/40.

| boot | p05 | p50 | p95 |
| --- | --- | --- | --- |
| n=40/40 | 0.399 | 0.455 | 0.571 |


## Extra — Y5 leftover after size (report only)

Y5 leftover after size rank 0.541 OLS 0.564 raw 0.561. Y5 never E — no AUROC as card X, leak_ok=False.

| slice | rank | OLS | ρ | raw | n |
| --- | --- | --- | --- | --- | --- |
| Y5 leftover after size (report only) | 0.541 | 0.564 | -0.913 | 0.561 | 4,905 |


## Extra — holdout coverage

Holdout 72 coverage only: 1,073 CM / 72 companies. open cov 54.2%; dark nn=0. No AUROC.

| slice | n_cm | companies | cov | dark_nn |
| --- | --- | --- | --- | --- |
| holdout 72 | 1,073 | 72 | 54.2% | 0 |


## Extra — open>0 leftover

open>0 leftover after days rank 0.553 OLS 0.694 ρ=-0.672 fake=False raw 0.483 days 0.725.

| slice | rank | OLS | ρ | raw | days | fake? |
| --- | --- | --- | --- | --- | --- | --- |
| open>0 leftover-days | 0.553 | 0.694 | -0.672 | 0.483 | 0.725 |  |


## Extra — log1p(open)

log1p(open) Y3 0.569 leftover-after-days rank 0.418 OLS 0.452 ρ=-0.017 fake=False.

| slice | rank | OLS | ρ | raw | fake? |
| --- | --- | --- | --- | --- | --- |
| log1p(open) leftover-days | 0.418 | 0.452 | -0.017 | 0.569 |  |


## Extra — leftover after pending

Leftover after e_pending_amt_share 0.562 fake=True; pending+days 0.469. Pending already decided — do not reopen.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| open leftover after pending | 0.562 | 0.522 | -0.818 | FALSE clone |
| leftover after pending+days | 0.469 | 0.537 | -0.802 | FALSE clone |


## Extra — SIZE terciles

SIZE terciles leftover after days: T1 0.408 T2+T3 0.560 T3 —.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| T1 | 0.408 | 0.646 | -0.672 |  | 847 |
| T2+T3 | 0.560 | 0.635 | -0.672 |  | 2,771 |
| T3 | — | — | -0.672 |  | 1,410 |


## Extra — company-median ρ

Company-median ρ open vs days 0.406 vs log1p(a_in3) 0.517 SIZE vs issued 0.769.

| pair | ρ | n |
| --- | --- | --- |
| company-median vs days | 0.406 | 744 |
| company-median vs log1p(a_in3) | 0.517 | 744 |
| company-median vs issued | 0.769 | 744 |


## Extra — so_far leftover after days

so_far leftover after days: short 0.512 mid 0.463 long —.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| short_<12 | 0.512 | 0.687 | -0.672 |  | 2,339 |
| mid_12_17 | 0.463 | 0.698 | -0.672 |  | 1,141 |
| long_>=18 | — | — | -0.672 |  | 138 |


## Extra — per-fold leftover

Per-fold rank leftover after days: 0.412 0.455 0.412 0.361 0.452.

| fold | rank leftover |
| --- | --- |
| 0 | 0.412 |
| 1 | 0.455 |
| 2 | 0.412 |
| 3 | 0.361 |
| 4 | 0.452 |


## Extra — leftover after issued+DPO / issued after open

Leftover after issued+DPO 0.531; after issued+DPO+days 0.540. Issued leftover after open 0.677 (issued CLOSED leftover 0.591).

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| open leftover after issued+DPO | 0.531 | 0.481 | 0.723 |
| open leftover after issued+DPO+days | 0.540 | 0.677 | 0.675 |
| issued leftover after open | 0.677 | 0.537 | -0.535 |


## Extra — Q6 mid / long

Q6 mid Y7 open_lag1 0.433 leftover 0.376; long Y7 0.464. Do not claim a TURNOVER seat.

| slice | Y7 open_lag1 | Y3 leftover days_lag1 | fake? |
| --- | --- | --- | --- |
| short_<12 | 0.478 | 0.442 |  |
| mid_12_17 | 0.433 | 0.376 |  |
| long_>=18 | 0.464 | — |  |


## Extra — KEEP-as-X scorecard

KEEP-as-X scorecard: beat size FAIL (Y3 0.569 < size 0.617); leftover-after-days FAIL 0.418 < 0.55; row not SIZE; gate twins PASS. Rewrite of issued. Do not put e_ap_open on the 15-col card.

| gate | value | pass? |
| --- | --- | --- |
| beat size ≥0.02 | -0.047 | FAIL |
| leftover after days (honest, ≥0.55, not fake) | 0.418 | FAIL |
| not SIZE |ρ| vs log1p(a_in3) <0.50 | 0.462 | PASS (row); company-median may SIZE |
| not twin vs days / n_tx / issued / DPO / pending | none | PASS (issued ρ=0.757 near) |
| leftover after issued (not a rewrite) | 0.459 | FAIL rewrite |


## Extra — days+size / intensity / Y7 / never-zero

Leftover after days+size 0.417; intensity 0.422; Y7 leftover-days 0.462; never-zero 0.593.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| leftover after days+size | 0.417 | 0.454 | 0.167 |
| open/days leftover after days | 0.422 | 0.666 | 0.854 |
| Y7 leftover after days | 0.462 | 0.454 | -0.672 |
| never-zero leftover-days | 0.593 | 0.705 | -0.672 |


## Extra — leftover after overdue

Leftover after e_ap_overdue 0.482 ρ_raw=-0.007. Overdue already decided — do not reopen.

| bar | rank | OLS | ρ | raw ρ |
| --- | --- | --- | --- | --- |
| open leftover after overdue | 0.482 | 0.603 | -0.911 | -0.007 |


## Extra — never-zero leftover after issued

never-zero leftover-days 0.593 fake=False raw 0.387; leftover-issued 0.637; issued+days 0.648; days-after-open 0.750.

| bar | rank | OLS | ρ | fake? | raw |
| --- | --- | --- | --- | --- | --- |
| never-zero leftover-days | 0.593 | 0.705 | -0.672 |  | 0.387 |
| never-zero leftover-issued | 0.637 | 0.389 | 0.712 |  | — |
| never-zero leftover issued+days | 0.648 | 0.701 | 0.038 |  | — |
| days leftover after open on never-zero | 0.750 | 0.734 | 0.348 |  | — |


## Extra — leftover after card KEEP leftovers

Leftover after c_ss_month 0.553; after c_salary_month 0.545; ss+salary+days 0.414. Stay off the 15-col card.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| leftover after c_ss_month | 0.553 | 0.742 | 0.808 |
| leftover after c_salary_month | 0.545 | 0.651 | 0.549 |
| leftover after ss+salary+days | 0.414 | 0.664 | 0.805 |


## Extra — DPO-defined leftover

DPO-defined leftover-days 0.552 fake=False; after DPO+days 0.429.

| bar | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| DPO-defined leftover-days | 0.552 | 0.691 | -0.672 |  | 3,015 |
| DPO-defined leftover DPO+days | 0.429 | 0.676 | 0.179 |  | 3,015 |


## Extra — ever-ERP leftover

ever-ERP leftover-days 0.418 fake=False; late 0.406.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| ever-ERP leftover-days | 0.418 | 0.695 | -0.672 |  | 3,618 |
| ever-ERP late | 0.406 | 0.701 | -0.672 |  | 2,952 |


## Extra — winsor / leftover after AR issued / AR-open+AP-issued

winsor leftover-days 0.418 fake=False; after AR issued 0.459; after AR-open+AP-issued 0.573.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| winsor99 leftover-days | 0.418 | 0.612 | -0.296 |  |
| leftover after e_ar_issued | 0.459 | 0.487 | -0.146 |  |
| leftover after AR open + AP issued | 0.573 | 0.548 | 0.077 |  |


## Extra — leftover after issued_lag1 (do not grow TURNOVER)

Y3 leftover after issued_lag1 0.447; Y7 0.430; after issued_lag1+days 0.458. Do not grow TURNOVER.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| Y3 leftover after issued_lag1 | 0.447 | 0.563 | 0.687 |
| Y7 leftover after issued_lag1 | 0.430 | 0.459 | 0.687 |
| Y3 leftover after issued_lag1+days | 0.458 | 0.660 | 0.185 |


## Extra — BETWEEN identity / leftover after own lag1

BETWEEN: leftover after company-mean 0.530 +days 0.563; after own lag1 0.589 +days 0.525; within leftover-days 0.615. ICC 0.99 BETWEEN — leftover is identity, not a new X.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| leftover after company-mean | 0.530 | 0.614 | -0.079 |
| leftover after company-mean+days | 0.563 | 0.578 | 0.154 |
| leftover after own lag1 | 0.589 | 0.593 | 0.011 |
| leftover after lag1+days | 0.525 | 0.549 | -0.067 |
| within (open−cmean) leftover-days | 0.615 | 0.578 | 0.569 |


## Extra — leftover after n_tx / joint days+issued+size / DPO after open

leftover after a_n_tx 0.422; n_tx+days 0.421; days+issued+size 0.550; DPO leftover after open 0.658. DPO already DROP — do not reopen.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| leftover after a_n_tx | 0.422 | 0.617 | 0.709 |
| leftover after n_tx+days | 0.421 | 0.688 | -0.446 |
| leftover after days+issued+size | 0.550 | 0.454 | 0.169 |
| DPO leftover after open | 0.658 | 0.626 | 0.489 |


## Extra — leftover after trail / AR_open_lag1

leftover after months_so_far 0.580 +days 0.419; after AR_open_lag1 0.423 +days 0.478. Not a trail-length rewrite; same-object AR lag dies leftover.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| leftover after months_so_far | 0.580 | 0.538 | -0.899 |
| leftover after trail+days | 0.419 | 0.463 | -0.893 |
| leftover after e_ar_open_lag1 | 0.423 | 0.552 | 0.095 |
| leftover after AR_open_lag1+days | 0.478 | 0.644 | 0.239 |


## Extra — leftover after log1p(issued) / issued+AR_open+days

leftover after log1p(issued) 0.459 +days 0.549; issued+AR_open+days 0.569. Rewrite of issued volume holds on log scale.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| leftover after log1p(issued) | 0.459 | 0.683 | -0.917 |
| leftover after log1p(issued)+days | 0.549 | 0.586 | -0.768 |
| leftover after issued+AR_open+days | 0.569 | 0.439 | 0.365 |


## What failed / next (held for wave note)

- Y3 leftover after days rank 0.418 OLS 0.695 fake=False
- inverse days leftover 0.720
- leftover after issued 0.459 rewrite=True
- leftover after DPO 0.605 numer=False
- Y5 leftover after size 0.541 leak_ok=False
- Q6 open_lag1 leftover 0.442 now 0.517
- leftover after AR open 0.423 same=True
- boot leftover p05=0.399 p50=0.455 p95=0.571
- open>0 leftover 0.553 fake=False
- log1p leftover 0.418 fake=False
- leftover after pending 0.562 +days 0.469 fake=True
- terciles T1 0.408 T2+T3 0.560
- co-median SIZE=True ρ=0.517
- so_far short 0.512 mid 0.463 long —
- per-fold leftover 0.412 0.455 0.412 0.361 0.452
- after issued+DPO 0.531 +days 0.540 issued-after-open 0.677
- Q6 mid Y7 0.433 leftover 0.376
- days+size leftover 0.417 intensity 0.422 Y7 0.462 never-zero 0.593
- leftover after overdue 0.482
- never-zero leftover-days 0.593 leftover-issued 0.637 +days 0.648
- leftover after ss 0.553 salary 0.545 ss+salary+days 0.414
- DPO-defined leftover 0.552 after DPO+days 0.429
- ever-ERP leftover 0.418 late 0.406
- winsor leftover 0.418 after AR issued 0.459 AR-open+AP-issued 0.573
- leftover after issued_lag1 Y3 0.447 Y7 0.430 +days 0.458
- BETWEEN leftover after cmean 0.530 +days 0.563 after lag1 0.589 within 0.615
- leftover after n_tx 0.422 +days 0.421 days+issued+size 0.550 DPO-after-open 0.658
- leftover after trail 0.580 +days 0.419 after AR_open_lag1 0.423 +days 0.478
- leftover after log1p(issued) 0.459 +days 0.549 issued+AR_open+days 0.569
- card: CLOSE unused leftover / KEEP off the 15-col card
- do not grow TURNOVER 0.720; do not put e_ap_open on the 15-col card

Elapsed 11s. Night quotes unchanged. Do not grow TURNOVER. Do not put e_ap_open on the 15-col card.
