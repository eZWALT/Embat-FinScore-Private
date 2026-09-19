# Unused leftover of `d_cust_lost` after `c_n_days_with_tx`

Generated `2026-09-19T07:50:01+02:00` by agent `e8b2c0d4`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_cust_lost`. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Do **not** grow TURNOVER. Do **not** put d_cust_lost on the 15-col Y3 card. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 never D. Y3 never B. Dark 470 stay NaN not 0. `d_n_cust` leftover 0.545 DROP — do not overwrite n_cust_qa. `d_cust_top1` leftover 0.525 DROP — do not overwrite top1_qa. Y4 HHI >0.975 footnote KEEP locked — do not overwrite y4_why / cust_hhi_qa.

`d_cust_lost` = |prev \ cur| of AR counterparties in the calendar quarter vs the previous quarter. Different object from Y7 `y7_top1_lost`. Feature report: 53.6% cov, acf1 0.58, ICC 0.98 BETWEEN.

## Headline

DROP leftover-after-days rank 0.522 (OLS 0.663, fake=False). Y3 lost 0.581 vs days 0.711 vs size 0.617 vs n_cust 0.653 vs top1 0.590. SIZE=False twin=True. Inverse days-after-lost 0.730. Leftover after n_cust 0.574 / top1 0.547 / new 0.463. Card: DROP from the 44 as Y3 X / CLOSE unused leftover. PARK as Y — do not invent y_cust_lost. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — do not invent y_cust_lost. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Q6 lag1 leftover after days_lag1 0.464. |
| 3 | Who is turning? | **DROP** leftover after days 0.522 vs days 0.711. |
| 4 | Dip vs fall? | Y4 leftover after top1 0.420 — footnote KEEP locked, do not overwrite y4_why. |
| 5 | Why did it change? | Twin screen: d_n_cust, d_cust_lost_lag1. ρ vs n_cust 0.861. |
| 6 | Months earlier? | lag1 leftover 0.464; days_lag1 0.684 (quote 0.684). |

## PARK / CLOSE / KEEP / DROP

| object | decision | why |
| --- | --- | --- |
| lost leftover after days (Y3 X) | **DROP** | Y3 leftover after days rank 0.522 OLS 0.663 dies; twin=True SIZE=False. Y3 0.581 vs days 0.711. n_cust leftover 0.545 DROP; top1 leftover 0.525 DROP. Stay off the 15-col card. Do not invent y_cust_lost. |
| 15-col Y3 card stem | **KEEP off the card** | do not put d_cust_lost on the card |
| TURNOVER add-on | **CLOSE** | do not grow 0.720; Y7 never D |
| d_n_cust leftover | **DROP (locked)** | rank 0.545 |
| d_cust_top1 leftover | **DROP (locked)** | rank 0.525 |
| Y4 HHI >0.975 footnote | **KEEP (locked)** | do not overwrite y4_why |
| health Y `y_cust_lost` | **PARK** | do not invent y_cust_lost |

## 1. Coverage / 470 vs ERP / SIZE ρ

Train d_cust_lost nn=11,338 cov=53.6% (feature report 53.6%). Dark never-ERP 470 (want 470): nn=0 zero=0 pos=0 CONFIRM NaN not 0. Ever-ERP 744 nn=11,338 of which zero=4,963. p50=1.000 p99=221.000 max=976.000. ρ vs log1p(a_in3)=0.322 n=10763 not SIZE (quote 0.323). acf1=0.582 (quote 0.58) ICC=0.975 (quote 0.98 BETWEEN) w/b=0.025 k=743.

| slice | n_cm | nn | cov | eq0 | p50 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| all train | 21,157 | 11,338 | 53.6% | 43.8% | 1.000 | 221.000 |
| ever-ERP | 13,554 | 11,338 | 83.7% | 43.8% | — | — |
| dark 470 | 7,603 | 0 | 0.0% | — | — | — |


## 2. Spearman twins / SIZE

cust_lost vs days ρ=0.417 (not a twin). vs a_n_tx 0.434 vs d_cust_new 0.657 vs d_n_cust 0.861 vs top1 -0.636 vs HHI -0.667 vs log1p(a_in3) 0.322 not SIZE. vs issued 0.480 vs own lag1 0.895. TWIN |ρ|≥0.80: d_n_cust, d_cust_lost_lag1. Gate twins: d_n_cust.

| vs | ρ | n | twin? |
| --- | --- | --- | --- |
| c_n_days_with_tx | 0.417 | 11,338 |  |
| a_n_tx | 0.434 | 11,338 |  |
| d_cust_new | 0.657 | 11,338 |  |
| d_n_cust | 0.861 | 10,978 | TWIN |
| d_cust_top1 | -0.636 | 8,779 |  |
| d_cust_hhi | -0.667 | 8,779 |  |
| log1p(a_in3) | 0.322 | 10,763 |  |
| e_ar_issued | 0.480 | 11,338 |  |
| d_cust_lost_lag1 | 0.895 | 10,595 | TWIN |


## 3. Single-feature group-fold Y3

Y3 d_cust_lost 0.581 vs days 0.711 (CONFIRM 0.711) vs size 0.617 (CONFIRM 0.617) vs d_n_cust 0.653 (CONFIRM 0.653) vs top1 0.590 (CONFIRM 0.590) vs new 0.634 vs HHI 0.595. Beat size -0.036.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_cust_lost | 3,080 | 221 | 0.581 | 0.057 | -1 | 0.650 0.625 0.514 0.579 0.538 |
| y3_recover_cash_6m | d_cust_lost_lag1 | 2,884 | 210 | 0.569 | 0.046 | -1 | 0.618 0.618 0.551 0.519 0.540 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | d_n_cust | 3,003 | 221 | 0.653 | 0.077 | -1 | 0.742 0.662 0.530 0.665 0.666 |
| y3_recover_cash_6m | d_cust_top1 | 2,485 | 141 | 0.590 | 0.116 | 1 | 0.738 0.411 0.588 0.607 0.606 |
| y3_recover_cash_6m | d_cust_hhi | 2,485 | 141 | 0.595 | 0.101 | 1 | 0.740 0.454 0.581 0.608 0.590 |
| y3_recover_cash_6m | d_cust_new | 3,080 | 221 | 0.634 | 0.089 | -1 | 0.768 0.595 0.541 0.677 0.590 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | e_ar_issued | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |


## 4. Honest leftover after days (Y3)

Y3 leftover after days rank 0.522 OLS 0.663 ρ(resid,days)=-0.536 R²=0.043 dies. Inverse: days leftover after lost rank 0.730 survives — keep the 0.711 bar.

| bar | rank | OLS | ρ | R² | fake? | dies? |
| --- | --- | --- | --- | --- | --- | --- |
| Y3 leftover after days | 0.522 | 0.663 | -0.536 | 0.043 |  | dies |
| inverse: days leftover after lost | 0.730 | 0.753 | 0.331 | 0.043 |  | lives |


## 5. Leftover after n_cust / top1 / HHI / new

Y3 leftover after d_n_cust 0.574 not just n_cust (n_cust leftover 0.545 DROP). After n_cust+days 0.553. After top1 0.547 +days 0.586 (top1 leftover 0.525 DROP). After HHI 0.551. After new 0.463 +days 0.549.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| leftover after d_n_cust | 0.574 | 0.642 | -0.371 |
| leftover after n_cust+days | 0.553 | 0.427 |  |
| leftover after d_cust_top1 | 0.547 | 0.605 | 0.579 |
| leftover after top1+days | 0.586 | 0.675 |  |
| leftover after d_cust_hhi | 0.551 | 0.608 | 0.519 |
| leftover after d_cust_new | 0.463 | 0.480 | -0.010 |
| leftover after new+days | 0.549 | 0.650 |  |


## 6. Q6 lag1 leftover after days_lag1

Q6 Y3 lost_lag1 on short 0.602. lag1 leftover after days_lag1 short rank 0.464 fake=False. now leftover after days_lag1 0.467. days_lag1 short 0.684 (quote 0.684). q6_keep = issued_lag1 / days_lag1 / ss_lag1. Do not claim a TURNOVER seat.

| slice | CV | n |
| --- | --- | --- |
| Q6 Y3 lost_lag1 short | 0.602 | 1,635 |
| Q6 leftover after days_lag1 short |  |  |
| Q6 now leftover after days_lag1 short |  |  |
| Q6 days_lag1 short | 0.684 |  |


## 7. Dark 470

Dark never-ERP 470 nn=0 zero=0 pos=0. CONFIRM NaN not 0.

## 8. Y7 leftover after issued (report only; never D)

Y7 leftover after issued is the issued leftover (report only). issued Y7 0.663 leftover-days 0.665. Y7 never D — d_cust_lost leak_ok=False. Do not grow TURNOVER 0.720.

| bar | CV | leftover-days | leak_ok |
| --- | --- | --- | --- |
| Y7 issued raw (report only) | 0.663 | 0.665 | False |


## 9. Bootstrap leftover-after-days

Bootstrap leftover-after-days rank p05=0.434 p50=0.511 p95=0.562 n=40/40.

| boot | p05 | p50 | p95 |
| --- | --- | --- | --- |
| n=40/40 | 0.434 | 0.511 | 0.562 |


## Extra — holdout coverage

Holdout 72 coverage only: 1,073 CM / 72 companies. lost cov 48.7%; dark nn=0. No AUROC.

| slice | n_cm | companies | cov | dark_nn |
| --- | --- | --- | --- | --- |
| holdout 72 | 1,073 | 72 | 48.7% | 0 |


## Extra — KEEP-as-X scorecard

KEEP-as-X scorecard: beat size FAIL (Y3 0.581 vs size 0.617); leftover-after-days FAIL 0.522; row not SIZE; gate twins FAIL. Do not put d_cust_lost on the 15-col card.

| gate | ok | value |
| --- | --- | --- |
| beat size ≥0.02 | FAIL | -0.036 |
| leftover after days ≥0.55 and not fake | FAIL | 0.522 |
| not SIZE |ρ|≥0.50 vs log1p(a_in3) | PASS | 0.322 |
| not twin vs days / n_tx / new / n_cust / top1 / HHI | FAIL | d_n_cust |


## Extra — SIZE terciles

SIZE terciles leftover after days: T1 0.575 T2+T3 0.587 T3 —.

| slice | rank | OLS | n |
| --- | --- | --- | --- |
| T1 | 0.575 | 0.643 | 713 |
| T2+T3 | 0.587 | 0.658 | 2,367 |
| T3 | — | — | 1,201 |


## Extra — so_far leftover after days

so_far leftover after days: short 0.474 mid 0.542 long —.

| slice | rank | OLS | n |
| --- | --- | --- | --- |
| short_<12 | 0.474 | 0.645 | 1,826 |
| mid_12_17 | 0.542 | 0.658 | 1,119 |
| long_>=18 | — | — | 135 |


## Extra — per-fold leftover

Per-fold rank leftover after days: — — — 0.541 0.582.

| fold | rank | OLS | n |
| --- | --- | --- | --- |
| 0 | — | — | 466 |
| 1 | — | — | 249 |
| 2 | — | — | 712 |
| 3 | 0.541 | 0.695 | 840 |
| 4 | 0.582 | 0.735 | 813 |


## Extra — BETWEEN identity

BETWEEN: leftover after company-mean 0.516; after own lag1 0.564; within leftover-days 0.551. ICC 0.98 BETWEEN.

| bar | rank | OLS |
| --- | --- | --- |
| leftover after company-mean | 0.516 | 0.560 |
| leftover after own lag1 | 0.564 | 0.494 |
| within leftover-days | 0.551 | 0.644 |


## Extra — same-n / lost>0

same-n Y3 lost 0.581 days 0.737 size 0.618 beat-size -0.037; lost>0 leftover-days 0.577.

| bar | CV | n |
| --- | --- | --- |
| Y3 lost same-n | 0.581 | 3,080 |
| Y3 days same-n | 0.737 |  |
| Y3 size same-n | 0.618 |  |
| lost>0 leftover-days |  |  |


## Extra — Y4 leftover (report; footnote KEEP locked)

Y4 leftover after top1 0.420; after days 0.446. Y4 HHI >0.975 footnote KEEP locked — do not overwrite y4_why.

| bar | rank | OLS |
| --- | --- | --- |
| Y4 leftover after top1 (report) | 0.420 | 0.567 |
| Y4 leftover after days (report) | 0.446 | 0.418 |


## Extra — leftover after days+size / card KEEP / issued / n_cust+top1

leftover after days+size 0.523; ss+salary+days 0.484; after issued 0.476 +days 0.552; n_cust+top1 0.595. Stay off the 15-col card.

| bar | rank | OLS |
| --- | --- | --- |
| leftover after days+size | 0.523 | 0.668 |
| leftover after ss+salary+days | 0.484 | 0.611 |
| leftover after e_ar_issued | 0.476 | 0.542 |
| leftover after issued+days | 0.552 | 0.665 |
| leftover after n_cust+top1 | 0.595 | 0.550 |


## Extra — company-median ρ

Company-median ρ lost vs days 0.448 vs log1p(a_in3) 0.317 not SIZE vs n_cust 0.896 TWIN vs new 0.853.

| vs | ρ | n |
| --- | --- | --- |
| days | 0.448 | 743 |
| log1p(a_in3) | 0.317 | 743 |
| d_n_cust | 0.896 | 743 |
| d_cust_new | 0.853 | 743 |


## Extra — Q6 mid / long

Q6 mid Y3 lost_lag1 0.532 leftover 0.545; long Y3 —. Do not claim a TURNOVER seat.

| slice | raw | leftover | n |
| --- | --- | --- | --- |
| Q6 Y3 mid_12_17 | 0.532 | 0.545 | 1,114 |
| Q6 Y3 long_>=18 | — | — | 135 |


## Extra — lost/n_cust rate / leftover after n_tx / new after lost

lost/n_cust raw 0.587 leftover-days 0.623; leftover after n_tx 0.519 +days 0.522; new leftover after lost 0.613.

| bar | CV | n |
| --- | --- | --- |
| lost/n_cust Y3 raw | 0.587 | 2,441 |
| lost/n_cust leftover-days |  |  |
| leftover after a_n_tx |  |  |
| leftover after n_tx+days |  |  |
| new leftover after lost |  |  |


## Extra — leftover after n_cust_lag1 / n_cust+new+days

leftover after n_cust_lag1 0.555 +days 0.540; n_cust+new+days 0.549. Twin of the customer book.

| bar | rank | OLS |
| --- | --- | --- |
| leftover after n_cust_lag1 | 0.555 | 0.627 |
| leftover after n_cust_lag1+days | 0.540 | 0.440 |
| leftover after n_cust+new+days | 0.549 | 0.482 |


## What failed / next (held for wave note)

- Y3 lost 0.581 vs days 0.711 vs size 0.617 beat=-0.036
- leftover-after-days rank 0.522 OLS 0.663 fake=False dies=True
- inverse days-after-lost 0.730
- leftover after n_cust 0.574 rewrite=False +days 0.553
- leftover after top1 0.547 +days 0.586 HHI 0.551 new 0.463
- Q6 lag1 leftover 0.464 days_lag1 0.684
- Y7 issued leftover-days 0.665 leak_ok=False
- boot leftover-days p05=0.434 p50=0.511 p95=0.562
- terciles T1 0.575 T2+T3 0.587
- so_far short 0.474 mid 0.542
- per-fold — — — 0.541 0.582
- BETWEEN cmean 0.516 lag1 0.564 within 0.551
- same-n lost 0.581 days 0.737 beat=-0.037 lost>0 0.577
- Y4 leftover-top1 0.420 leftover-days 0.446
- leftover after days+size 0.523 card 0.484 issued 0.476 n_cust+top1 0.595
- company-median ρ vs n_cust 0.896 twin=True
- Q6 mid leftover 0.545 long raw —
- lost/n_cust leftover-days 0.623 raw 0.587 leftover-ntx 0.519 new-after-lost 0.613
- leftover after n_cust_lag1 0.555 +days 0.540 n_cust+new+days 0.549
- card: DROP from the 44 as Y3 X / CLOSE unused leftover
- do not grow TURNOVER 0.720; do not put d_cust_lost on the 15-col card; PARK y_cust_lost
