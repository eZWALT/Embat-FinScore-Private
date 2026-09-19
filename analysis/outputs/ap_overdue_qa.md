# Unused leftover of `e_ap_overdue` after `c_n_days_with_tx`

Generated `2026-09-19T07:18:26+02:00` by agent `e8b2c0d4`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_ap_overdue`. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Do **not** grow TURNOVER. Do **not** put e_ap_overdue on the 15-col Y3 card. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. AP open leftover 0.418 CLOSED — do not overwrite ap_open_qa. Delay leftover 0.427 CLOSE / DROP as Y3 X — do not overwrite delay_qa. DPO already DROP — do not overwrite dpo_qa.

`e_ap_overdue` = overdue AP |amount| / open (due < period end). Feature report: 57.5% cov, acf1 0.51, ICC 0.95 BETWEEN. `e_ap_overdue_30` twin |ρ| 0.85.

## Headline

CLOSE leftover-after-days rank 0.584 (OLS 0.564, fake=False). Y3 overdue 0.625 vs days 0.711 vs size 0.617 vs open 0.569 vs delay_paid 0.521 vs DPO 0.627. SIZE=False twin=False. Inverse days-after-overdue 0.711. Leftover after open 0.623. Leftover after DPO 0.544 / delay 0.579. Card: CLOSE unused leftover / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## PARK / CLOSE / KEEP / DROP

| object | decision | why |
| --- | --- | --- |
| overdue leftover after days (Y3 X) | **CLOSE** | Y3 0.625 vs days 0.711 leftover 0.584. Does not clear KEEP-as-X. Stay off the card. |
| 15-col Y3 card stem | **KEEP off the card** | do not put e_ap_overdue on the card |
| TURNOVER add-on | **CLOSE** | do not grow 0.720 |
| AP open leftover | **CLOSE (locked)** | rank 0.418 |
| delay / overdue as Y3 X | **DROP (locked delay_qa)** | Y3 0.512 leftover 0.427 |
| DPO | **DROP (locked)** | do not reopen dpo_qa |
| health Y `y_ap_overdue` | **PARK** | do not invent y_ap_overdue |

## 1. Coverage / 470 vs ERP / SIZE ρ

Train e_ap_overdue nn=12,173 cov=57.5% (feature report 57.5%). Dark never-ERP 470 (want 470): nn=0 zero=0 pos=0 CONFIRM NaN not 0. Ever-ERP 744 nn=12,173 of which zero=1,461. p50=0.680 p99=1.000 max=1.000. ρ vs log1p(a_in3)=-0.229 n=11065 not SIZE (quote -0.202). acf1=0.508 (quote 0.51) ICC=0.954 (quote 0.95 BETWEEN) w/b=0.048 k=740.

| slice | n_cm | nn | cov | eq0 | p50 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| all train | 21,157 | 12,173 | 57.5% | 12.0% | 0.680 | 1.000 |
| ever-ERP | 13,554 | 12,173 | 89.8% | 12.0% | — | — |
| dark 470 | 7,603 | 0 | 0.0% | — | — | — |


## 2. Spearman twins / SIZE

ap_overdue vs days ρ=-0.183 (not a twin). vs a_n_tx -0.195 vs e_ap_open -0.007 vs DPO 0.500 vs delay_paid 0.329 vs log1p(a_in3) -0.229 not SIZE. vs overdue_30 0.849 vs AR overdue 0.465 vs own lag1 0.814. TWIN |ρ|≥0.80: e_ap_overdue_30, e_ap_overdue_lag1. Gate twins: none.

| vs | ρ | n | twin? |
| --- | --- | --- | --- |
| c_n_days_with_tx | -0.183 | 12,173 |  |
| a_n_tx | -0.195 | 12,173 |  |
| e_ap_open | -0.007 | 12,173 |  |
| e_dpo_proxy | 0.500 | 10,654 |  |
| e_delay_paid | 0.329 | 8,334 |  |
| log1p(a_in3) | -0.229 | 11,065 |  |
| e_ap_overdue_30 | 0.849 | 12,173 | TWIN |
| e_ar_overdue | 0.465 | 9,580 |  |
| e_ap_overdue_lag1 | 0.814 | 11,340 | TWIN |


## 3. Single-feature group-fold Y3 / Y7

Y3 e_ap_overdue 0.625 vs days 0.711 (CONFIRM 0.711) vs size 0.617 (CONFIRM 0.617) vs e_ap_open 0.569 (CONFIRM 0.569) vs delay_paid 0.521 vs DPO 0.627 vs overdue_30 0.620. Beat size 0.008. Y7 overdue 0.407.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | e_ap_overdue | 3,336 | 223 | 0.625 | 0.126 | 1 | 0.772 0.561 0.520 0.750 0.520 |
| y3_recover_cash_6m | e_ap_overdue_lag1 | 3,285 | 217 | 0.609 | 0.123 | 1 | 0.750 0.527 0.528 0.737 0.505 |
| y3_recover_cash_6m | e_ap_overdue_30 | 3,336 | 223 | 0.620 | 0.142 | 1 | 0.787 0.548 0.491 0.759 0.515 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | e_ap_open | 3,618 | 264 | 0.569 | 0.109 | -1 | 0.638 0.674 0.461 0.443 0.632 |
| y3_recover_cash_6m | e_delay_paid | 2,294 | 150 | 0.521 | 0.098 | 1 | 0.419 0.441 0.543 0.665 0.536 |
| y3_recover_cash_6m | e_dpo_proxy | 3,015 | 174 | 0.627 | 0.112 | 1 | 0.797 0.510 0.605 0.669 0.554 |
| y3_recover_cash_6m | e_ar_overdue | 2,655 | 143 | 0.616 | 0.108 | 1 | 0.776 0.591 0.558 0.661 0.493 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y7_top1_lost | e_ap_overdue | 7,278 | 2,086 | 0.407 | 0.064 | 1 | 0.494 0.376 0.453 0.344 0.365 |
| y7_top1_lost | e_ap_overdue_lag1 | 7,009 | 1,987 | 0.416 | 0.071 | -1 | 0.510 0.376 0.473 0.362 0.358 |
| y7_top1_lost | e_ap_overdue_30 | 7,278 | 2,086 | 0.416 | 0.069 | -1 | 0.489 0.370 0.495 0.370 0.356 |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 |
| y7_top1_lost | log1p_a_in3 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 |
| y7_top1_lost | e_ap_open | 7,464 | 2,149 | 0.460 | 0.081 | -1 | 0.527 0.410 0.530 0.488 0.343 |
| y7_top1_lost | e_delay_paid | 5,591 | 1,522 | 0.444 | 0.096 | -1 | 0.518 0.438 0.562 0.342 0.358 |
| y7_top1_lost | e_dpo_proxy | 7,038 | 1,929 | 0.450 | 0.051 | -1 | 0.497 0.403 0.510 0.438 0.403 |
| y7_top1_lost | e_ar_overdue | 6,859 | 1,912 | 0.605 | 0.048 | 1 | 0.582 0.668 0.540 0.630 0.603 |
| y7_top1_lost | a_n_tx | 7,464 | 2,149 | 0.452 | 0.032 | -1 | 0.461 0.482 0.476 0.436 0.405 |


## 4. Honest leftover after days (Y3)

Y3 leftover after days rank 0.584 OLS 0.564 ρ(resid,days)=0.072 R²=0.016 lives. Inverse: days leftover after overdue rank 0.711 survives — keep the 0.711 bar.

| bar | rank | OLS | ρ | R² | fake? | dies? |
| --- | --- | --- | --- | --- | --- | --- |
| Y3 leftover after days | 0.584 | 0.564 | 0.072 | 0.016 |  | lives |
| inverse: days leftover after overdue | 0.711 | 0.717 | -0.066 | 0.016 |  | lives |


## 5. Leftover after e_ap_open / DPO / delay_paid

Y3 leftover after e_ap_open rank 0.623 OLS 0.626 ρ=-0.053 not just the open stock. After open+days 0.581 fake=False. Open leftover after overdue 0.482 (open CLOSE leftover 0.418).

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| leftover after e_ap_open | 0.623 | 0.626 | -0.053 |  |
| leftover after open+days | 0.581 | 0.565 | 0.092 |  |
| open leftover after overdue | 0.482 | 0.603 | -0.911 | FALSE clone |


Y3 leftover after DPO rank 0.544; after DPO+days 0.525. After delay_paid 0.579; after delay+days 0.549. DPO DROP locked; delay leftover CLOSE (Y3 0.512 leftover 0.427). Do not overwrite delay_qa / dpo_qa.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| leftover after DPO | 0.544 | 0.589 | 0.474 |  |
| leftover after DPO+days | 0.525 | 0.566 | 0.493 |  |
| leftover after delay_paid | 0.579 | 0.544 | 0.061 |  |
| leftover after delay+days | 0.549 | 0.447 | 0.067 |  |


## 6. vs e_ap_overdue_30

vs overdue_30: leftover 0.567 (twin quote 0.85); inverse 0.575; after 30+days 0.438; overdue_30 leftover-days 0.590.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| overdue leftover after overdue_30 | 0.567 | 0.434 | 0.369 |  |
| overdue_30 leftover after overdue | 0.575 | 0.638 | 0.396 |  |
| overdue leftover after 30+days | 0.438 | 0.452 | 0.365 |  |
| overdue_30 leftover after days | 0.590 | 0.576 | 0.083 |  |


## 7. Dark 470

Dark never-ERP 470 nn=0 zero=0 pos=0. CONFIRM NaN not 0.

## 8. Q6 lag1 leftover after days_lag1

Q6 Y7 overdue_lag1 on short 0.423. lag1 leftover after days_lag1 short rank 0.433 fake=False. Y3 now leftover after days_lag1 short 0.557. Y3 lag1 leftover 0.458. Delay first-6 nn=0 (delay Q6 CLOSE empty until month 7). Do not claim a TURNOVER seat.

| slice | CV | n | n_pos |
| --- | --- | --- | --- |
| Q6 Y7 overdue_lag1 short | 0.423 | 4,014 | 1,192 |
| Q6 Y7 lag1 leftover after days_lag1 short |  |  |  |
| Q6 Y3 now leftover after days_lag1 short |  |  |  |
| Q6 Y3 lag1 leftover after days_lag1 short |  |  |  |


## 9. Y5 leftover after size (report only)

Y5 leftover after size rank 0.427 OLS 0.433 raw 0.427. Y5 never E — no AUROC as card X, leak_ok=False.

| bar | rank | OLS | raw | leak_ok |
| --- | --- | --- | --- | --- |
| Y5 leftover after size (report only) | 0.427 | 0.433 | 0.427 | False |


## 10. Bootstrap leftover-after-days

Bootstrap leftover-after-days rank p05=0.420 p50=0.579 p95=0.632 n=40/40.

| boot | p05 | p50 | p95 |
| --- | --- | --- | --- |
| n=40/40 | 0.420 | 0.579 | 0.632 |


## Extra — holdout coverage

Holdout 72 coverage only: 1,073 CM / 72 companies. overdue cov 49.2%; dark nn=0. No AUROC.

| slice | n_cm | companies | cov | dark_nn |
| --- | --- | --- | --- | --- |
| holdout 72 | 1,073 | 72 | 49.2% | 0 |


## Extra — SIZE terciles

SIZE terciles leftover after days: T1 0.476 T2+T3 0.559 T3 —.

| slice | rank | OLS | n |
| --- | --- | --- | --- |
| T1 | 0.476 | 0.504 | 751 |
| T2+T3 | 0.559 | 0.538 | 2,585 |
| T3 | — | — | 1,338 |


## Extra — company-median ρ

Company-median ρ overdue vs days -0.201 vs log1p(a_in3) -0.227 not SIZE vs open -0.034 vs delay 0.408.

| vs | ρ | n |
| --- | --- | --- |
| days | -0.201 | 740 |
| log1p(a_in3) | -0.227 | 740 |
| e_ap_open | -0.034 | 740 |
| e_delay_paid | 0.408 | 683 |


## Extra — so_far leftover after days

so_far leftover after days: short 0.526 mid 0.598 long —.

| slice | rank | OLS | n |
| --- | --- | --- | --- |
| short_<12 | 0.526 | 0.546 | 2,104 |
| mid_12_17 | 0.598 | 0.579 | 1,100 |
| long_>=18 | — | — | 132 |


## Extra — per-fold leftover

Per-fold rank leftover after days: — — — 0.693 0.469.

| fold | rank | OLS | n |
| --- | --- | --- | --- |
| 0 | — | — | 487 |
| 1 | — | — | 278 |
| 2 | — | — | 762 |
| 3 | 0.693 | 0.652 | 926 |
| 4 | 0.469 | 0.455 | 883 |


## Extra — Q6 mid / long

Q6 mid Y7 overdue_lag1 0.380 leftover 0.379; long Y7 0.505. Delay Q6 CLOSE. Do not claim a TURNOVER seat.

| slice | raw | leftover | n |
| --- | --- | --- | --- |
| Q6 Y7 mid_12_17 | 0.380 | 0.379 | 2,088 |
| Q6 Y7 long_>=18 | 0.505 | 0.502 | 907 |


## Extra — KEEP-as-X scorecard

KEEP-as-X scorecard: beat size FAIL (Y3 0.625 vs size 0.617); leftover-after-days PASS 0.584; row not SIZE; gate twins PASS. Do not put e_ap_overdue on the 15-col card.

| gate | ok | value |
| --- | --- | --- |
| beat size ≥0.02 | FAIL | 0.008 |
| leftover after days ≥0.55 and not fake | PASS | 0.584 |
| not SIZE |ρ|≥0.50 vs log1p(a_in3) | PASS | -0.229 |
| not twin vs days / n_tx / open / DPO / delay_paid | PASS | none |


## Extra — ever-ERP / late leftover

ever-ERP leftover-days 0.584 fake=False; late 0.583.

| slice | rank | OLS |
| --- | --- | --- |
| ever-ERP leftover-days | 0.584 | 0.564 |
| late (month≥7) leftover-days | 0.583 | 0.556 |


## Extra — BETWEEN identity

BETWEEN: leftover after company-mean 0.579; after own lag1 0.600; within leftover-days 0.483. ICC 0.95 BETWEEN.

| bar | rank | OLS |
| --- | --- | --- |
| leftover after company-mean | 0.579 | 0.572 |
| leftover after own lag1 | 0.600 | 0.583 |
| within leftover-days | 0.483 | 0.566 |


## Extra — Y7 leftover (do not grow TURNOVER)

Y7 leftover-days 0.406; leftover-open 0.404. Do not grow TURNOVER 0.720.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| Y7 leftover after days | 0.406 | 0.413 | 0.072 |
| Y7 leftover after open | 0.404 | 0.410 |  |


## Extra — leftover after days+size / card KEEP / AR overdue / open+DPO+delay

leftover after days+size 0.589; ss+salary+days 0.568; after AR overdue 0.603 +days 0.576; open+DPO+delay 0.452. Stay off the 15-col card.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| leftover after days+size | 0.589 | 0.582 | 0.056 |
| leftover after ss+salary+days | 0.568 | 0.487 |  |
| leftover after e_ar_overdue | 0.603 | 0.601 | 0.073 |
| leftover after AR overdue+days | 0.576 | 0.557 |  |
| leftover after open+DPO+delay | 0.452 | 0.534 |  |


## Extra — overdue>0 / winsor / open-defined

overdue>0 leftover-days 0.624; winsor 0.584; open-defined 0.584.

| bar | rank | OLS | n |
| --- | --- | --- | --- |
| overdue>0 leftover-days | 0.624 | 0.601 | 2,945 |
| winsor99 leftover-days | 0.584 | 0.564 |  |
| open-defined leftover-days | 0.584 | 0.564 |  |


## Extra — same-n days/size / leftover after n_tx / trail

same-n Y3 overdue 0.625 days 0.725 size 0.599 beat-size 0.026; leftover after n_tx 0.582 +days 0.582; trail 0.617 +days 0.583.

| bar | CV | n | n_pos |
| --- | --- | --- | --- |
| Y3 overdue same-n | 0.625 | 3,336 | 223 |
| Y3 days same-n | 0.725 | 3,336 | 223 |
| Y3 size same-n | 0.599 | 3,277 | 217 |
| leftover after a_n_tx |  |  |  |
| leftover after n_tx+days |  |  |  |
| leftover after months_so_far |  |  |  |
| leftover after trail+days |  |  |  |


## Extra — leftover after delay_lag1 / DPO-defined

leftover after delay_lag1 0.573 +days 0.543; DPO-defined leftover-days 0.547 DPO+days 0.525; rank leftover sign=1. Delay Q6 CLOSE — do not grow TURNOVER.

| bar | rank | OLS |
| --- | --- | --- |
| leftover after delay_lag1 | 0.573 | 0.544 |
| leftover after delay_lag1+days | 0.543 | 0.426 |
| DPO-defined leftover-days | 0.547 | 0.543 |
| DPO-defined leftover DPO+days | 0.525 | 0.566 |
| leftover rank sign | 0.584 |  |


## Extra — leftover after open_lag1 / days+n_tx+size

leftover after open_lag1 0.624 +days 0.574; days+n_tx+size 0.586. Stay off the 15-col card.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| leftover after open_lag1 | 0.624 | 0.625 | -0.007 |
| leftover after open_lag1+days | 0.574 | 0.561 |  |
| leftover after days+n_tx+size | 0.586 | 0.581 |  |


## What failed / next (held for wave note)

- Y3 overdue 0.625 vs days 0.711 vs size 0.617 beat=0.008
- leftover-after-days rank 0.584 OLS 0.564 fake=False dies=False
- inverse days-after-overdue 0.711
- leftover after open 0.623 rewrite=False +days 0.581
- leftover after DPO 0.544 +days 0.525 after delay 0.579 +days 0.549
- vs overdue_30 leftover 0.567 leftover-days-30 0.590
- Q6 Y7 lag1 leftover 0.433 Y3 now 0.557 delay_early_nn=0
- Y5 leftover-size 0.427 leak_ok=False
- boot leftover-days p05=0.420 p50=0.579 p95=0.632
- terciles T1 0.476 T2+T3 0.559
- so_far short 0.526 mid 0.598
- per-fold — — — 0.693 0.469
- Q6 mid leftover 0.379 long raw 0.505
- ever-ERP leftover 0.584 late 0.583
- BETWEEN cmean 0.579 lag1 0.600 within 0.483
- Y7 leftover-days 0.406 leftover-open 0.404
- leftover after days+size 0.589 ss+salary+days 0.568 AR 0.603 triple 0.452
- overdue>0 leftover 0.624 winsor 0.584 open-defined 0.584
- same-n overdue 0.625 days 0.725 size 0.599 beat=0.026 leftover-ntx 0.582 trail+days 0.583
- leftover after delay_lag1 0.573 +days 0.543 DPO-defined 0.547 DPO+days 0.525 sign=1
- leftover after open_lag1 0.624 +days 0.574 days+n_tx+size 0.586
- card: CLOSE unused leftover / KEEP off the 15-col card
- do not grow TURNOVER 0.720; do not put e_ap_overdue on the 15-col card
