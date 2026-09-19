# Unused leftover of `f_util_snapshot` after `c_n_days_with_tx`

Generated `2026-09-19T07:13:23+02:00` by agent `b17e9c44`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. Do not invent a Y. Do not put util on the 15-col card. Do not overwrite `debt_schedule_qa` / `ogtg_qa.*` / `factoring_qa.*` / `n_types_qa.*` / `ar_open_qa.*` / `ap_open_qa.*`. Y3 never B. Night Y3 **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 TURNOVER **0.720 / 0.712**.

`f_util_snapshot` = sum|outstanding| / sum|granted| over as-of facilities. NaN until the 2026-08 extract; NaN on last-month companies with no facilities. NORTH_STAR PARK snapshot cols as X. Utilisation is impossible as a Y (Y10 PARK). `f_outstanding_gt_granted` just PARK leftover 0.540.

## Headline

PARK as snapshot X. Leftover after days **CLOSE** native n_pos=0 hole-rank 0.711 OLS 0.711. Last-month-only=True cov 1.6% vs OGTG 5.7%. Y3 native — vs days 0.711 vs size 0.617 vs n_types 0.578. Twin=none ρ days 0.034 types -0.041 OGTG 0.152. Rise/extract True n=334. Q6 CLOSE. Y10 impossible. PARK as snapshot / DROP from the 44 as Y3 X. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged. Do not put util on the 15-col card.

## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| util leftover after days | **CLOSE** | native n_pos=0 hole-rank 0.711 OLS 0.711 fake=True |
| as Y3 X (not on 15-col card) | **PARK as snapshot / DROP from the 44 as Y3 X** | Last-month extract (cov 1.6%; Y3 n_pos on snapshot=0). Snapshot cannot lead (Q6 CLOSE). Utilisation impossible as a Y (Y10 PARK). Hole leftover after days 0.711. Do not put on the 15-col card. |
| Last-month-only / snapshot | **YES** | Train util defined 334/21,157 (1.6%; quote 1.6% CONFIRM). OGTG defined 5.7% (quote 5.7% CONFIRM). Defined periods ['2026-08'] last-month-only=True last-share 100.0% (last 2026-08 n=334; quote 334 CONFIRM). median 0.499 p95 1.000 zeros 36. ρ vs log1p(a_in3) -0.016 (not SIZE). acf1 —. |
| Twin / SIZE | twin=no SIZE=no | ρ days 0.034 types -0.041 OGTG 0.152 size -0.016 |
| Q6 lag leftover | **CLOSE** | Snapshot cannot lead: native util Y3 n_pos=0 (LOW_POWER on extract month). fillna0 lag1 leftover after days_lag1 0.632 ρ=— fake=False. Days lag1 0.684 (KEEP 0.684 CONFIRM). Q6 CLOSE. |
| Rise / extract hole | **extract-hole** | Util rises 0 / drops 0 / defined diffs 0 (extract-hole: no within-company pair). Last-month defined 334 (quote 334 CONFIRM). Y10 utilisation impossible=CONFIRM (no outstanding/granted history; snapshot last-month only). |
| as health Y / Y10 | **PARK / impossible** | Y10 utilisation impossible CONFIRM: 1/24 months have any util (no outstanding/granted history; snapshot last-month only). Do not add y10 to FROZEN_ACCEPTED. Do not invent a utilisation Y. |
| Night quotes | **unchanged** | Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712 |


## 1. Coverage — last-month vs OGTG

Train util defined 334/21,157 (1.6%; quote 1.6% CONFIRM). OGTG defined 5.7% (quote 5.7% CONFIRM). Defined periods ['2026-08'] last-month-only=True last-share 100.0% (last 2026-08 n=334; quote 334 CONFIRM). median 0.499 p95 1.000 zeros 36. ρ vs log1p(a_in3) -0.016 (not SIZE). acf1 —.

Spearman twins |ρ|≥0.80 on defined util: none. vs days 0.034 vs a_n_tx 0.041 vs n_types -0.041 vs OGTG 0.152 vs size -0.016.

| vs | ρ | twin |
| --- | --- | --- |
| c_n_days_with_tx | 0.034 |  |
| a_n_tx | 0.041 |  |
| f_n_types | -0.041 |  |
| f_outstanding_gt_granted | 0.152 |  |
| log1p(a_in3) | -0.016 |  |
| f_n_facilities | 0.001 |  |


## 2. Single-feature train group-fold AUROC

Y3 native util — n=0 n_pos=0 (LOW_POWER — snapshot month has no Y3). fillna0 hole 0.500 defined-dummy 0.500. vs days 0.711 (0.711 CONFIRM) vs size 0.617 (0.617 CONFIRM) vs n_types 0.578 (0.578 CONFIRM) vs OGTG —.

| y | feature | n | n_pos | CV | sign | folds |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | f_util_snapshot | 0 | 0 | LOW_POWER | — | — |
| y3_recover_cash_6m | util_fillna0 | 5,648 | 402 | 0.500 | 1 | 0.500 0.500 0.500 0.500 0.500 |
| y3_recover_cash_6m | util_defined_dummy | 5,648 | 402 | 0.500 | 1 | 0.500 0.500 0.500 0.500 0.500 |
| y3_recover_cash_6m | f_outstanding_gt_granted | 0 | 0 | LOW_POWER | — | — |
| y3_recover_cash_6m | f_n_types | 5,648 | 402 | 0.578 | -1 | 0.638 0.574 0.630 0.504 0.543 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y2_neg_2of3 | f_util_snapshot | 0 | 0 | LOW_POWER | — | — |
| y2_neg_2of3 | util_fillna0 | 17,356 | 1,271 | 0.500 | 1 | 0.500 0.500 0.500 0.500 0.500 |
| y2_neg_2of3 | util_defined_dummy | 17,356 | 1,271 | 0.500 | 1 | 0.500 0.500 0.500 0.500 0.500 |
| y2_neg_2of3 | f_outstanding_gt_granted | 0 | 0 | LOW_POWER | — | — |
| y2_neg_2of3 | f_n_types | 17,356 | 1,271 | 0.523 | 1 | 0.541 0.539 0.446 0.563 0.524 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | log1p(a_in3) | 14,968 | 1,044 | 0.552 | 1 | 0.523 0.609 0.595 0.520 0.513 |


## 3. Honest leftover after days + inverse + OGTG / n_types

Native leftover after days rank — n_pos=0 (LOW_POWER). Last-month-rows leftover — n_pos=0. fillna0 hole leftover after days rank 0.711 OLS 0.711 ρ(resid,days)=-0.959 fake=True. Defined-dummy leftover 0.711 fake=True. After n_types hole 0.578 after OGTG hole 0.500. Inverse days after hole 0.711. Honest leftover DIES.

| control | OLS | rank | n | n_pos | ρ(resid,ctrl) | fake | dies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native after days (last-month rows) | — | — | 0 | 0 | 0.070 |  | YES |
| native after days (Y3 panel) | — | — | 0 | 0 | 0.070 |  | YES |
| hole after days | 0.711 | 0.711 | 5648 | 402 | -0.959 | YES | YES |
| dummy after days | 0.711 | 0.711 | 5648 | 402 | -0.955 | YES | YES |
| hole after n_types | 0.578 | 0.578 | 5648 | 402 | -0.880 | YES | YES |
| hole after OGTG-fillna0 | 0.500 | 0.500 | 5648 | 402 | -0.104 |  | YES |
| dummy after n_types | 0.578 | 0.578 | 5648 | 402 | -0.859 | YES | YES |
| dummy after OGTG-fillna0 | 0.500 | 0.500 | 5648 | 402 | 0.225 |  | YES |
| hole after size | 0.617 | 0.617 | 5528 | 391 | -0.954 | YES | YES |
| days after native util (inverse) | — | — | 0 | 0 | 0.082 |  | YES |
| days after hole fillna0 (inverse) | 0.711 | 0.711 | 5648 | 402 | 0.027 |  | no |


## 4. Rise-only / extract-hole / Y10 impossible

Util rises 0 / drops 0 / defined diffs 0 (extract-hole: no within-company pair). Last-month defined 334 (quote 334 CONFIRM). Y10 utilisation impossible=CONFIRM (no outstanding/granted history; snapshot last-month only).

Y10 utilisation impossible CONFIRM: 1/24 months have any util (no outstanding/granted history; snapshot last-month only). Do not add y10 to FROZEN_ACCEPTED. Do not invent a utilisation Y.

## 5. Dark vs ERP leftover

Last-month ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). util defined invoiced 198 dark 136. Hole leftover after days invoiced 0.730 dark 0.704.

| slice | n | defined | rate | median |
| --- | --- | --- | --- | --- |
| invoiced_744 | 744 | 198 | 26.6% | 0.510 |
| dark_470 | 470 | 136 | 28.9% | 0.498 |


## 6. Q6 — snapshot cannot lead

Snapshot cannot lead: native util Y3 n_pos=0 (LOW_POWER on extract month). fillna0 lag1 leftover after days_lag1 0.632 ρ=— fake=False. Days lag1 0.684 (KEEP 0.684 CONFIRM). Q6 CLOSE.

| col | n | n_pos | CV |
| --- | --- | --- | --- |
| f_util_snapshot | 0 | 0 | LOW_POWER |
| util0 | 5,648 | 402 | 0.500 |
| util0_lag1 | 5,648 | 402 | 0.500 |
| c_n_days_with_tx | 5,648 | 402 | 0.711 |
| c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 |


## 7. Last-month trait

Company last-month util defined 334/1214. Trait leftover after days 0.622 ρ=0.044 fake=False raw 0.633. After n_types 0.634 after OGTG 0.633. Ever-Y3 recover util 28/334 (12.1%) vs rest 29.6%.

Last-month labeled Y3 0 Y2 0 (horizon → empty; quote Y3=0). Util defined 334.

## Extra — holdout coverage

Holdout coverage only (no fit): 72 co / 1,073 CM, defined 13 last-month 13 periods ['2026-08'] last-month-only=True (debt-schedule quote 13 CONFIRM True).

## Extra — OGTG overlap

Last-month util defined 334 OGTG=1 32 both 28. ρ util~OGTG 0.152 (OGTG leftover after days just PARK 0.540).

## Extra — last-month util by SIZE tercile

Last-month util defined by SIZE tercile: T1 12.6% med 0.389, T2 25.5% med 0.575, T3 44.4% med 0.498

| tercile | n | defined | rate | median |
| --- | --- | --- | --- | --- |
| T1 | 405 | 51 | 12.6% | 0.389 |
| T2 | 404 | 103 | 25.5% | 0.575 |
| T3 | 405 | 180 | 44.4% | 0.498 |


## Extra — trait leftover stacks

Trait leftover after days+n_types 0.628 ρ=0.022; after days+OGTG 0.622; after days+size 0.625.

Util-defined company dummy leftover after days 0.458 fake=False. Trait leftover after defined-dummy 0.633; after days+dummy 0.622.

Trait leftover-after-days rank folds 0.745 0.572 0.610 0.424 0.760 rank 0.622 ρ=0.044 fake=False.

Plot: `util_snap_qa.png`.

## What failed / next

- no replica miss; native leftover undefined (Y3 n_pos=0 on 2026-08). fillna0 / defined-dummy leftover 0.711 is a fake days leak (ρ=-0.959). Trait leftover 0.622 is not a days leak (ρ=0.044) but folds 0.424–0.760 / boot p05 0.391 dies. PARK snapshot X. Utilisation impossible as a Y.

## Later extras (same module)

- Company bootstrap trait leftover-after-days n=44 p05/p50/p95 0.391 / 0.607 / 0.693.
- Last-month util defined 334 OGTG=1 32 both 28. ρ util~OGTG 0.152 (OGTG leftover after days just PARK 0.540).
- Trait leftover after days+n_types 0.628 ρ=0.022; after days+OGTG 0.622; after days+size 0.625.
- Util-defined company dummy leftover after days 0.458 fake=False. Trait leftover after defined-dummy 0.633; after days+dummy 0.622.
- Trait leftover after days invoiced — ρ=0.044 dark — ρ=0.044 (access ≠ ERP).
- Permute last-month util leftover-after-days null p50 0.479 p(obs≥null)=0.042 obs=0.622.
- Trait leftover-after-days rank folds 0.745 0.572 0.610 0.424 0.760 rank 0.622 ρ=0.044 fake=False.
- Company-level last-month util vs ever-Y3 AUROC + 0.419 − 0.581 n=232 n_pos=28.
- Trait leftover after last-month days 0.614 ρ=0.022; after last-month days+n_types 0.617 (above 0.55).
- Y2 trait leftover after days 0.558 ρ=0.044 fake=False raw 0.558.
- Inverse: days leftover after util-trait 0.730; n_types leftover after util-trait 0.310 (0.711 bar should live).
- Trait leftover after days T1/T2/T3 — / — / —.
- Q6 fillna0 lag1 leftover after contemporaneous days 0.500 ρ=— fake=False; lag3 after days 0.500 (snapshot lag cannot lead).
- Trait leftover after last-month has_loc 0.630; after last-month n_facilities 0.636; after last-month days+n_types+has_loc 0.616.
- Trait leftover-after-last-month days+n_types rank folds 0.739 0.573 0.597 0.408 0.767 rank 0.617.
- BETWEEN leftover of util-trait after days-mean 0.624 ρ=0.058 fake=False.
- Last-month util defined by SIZE tercile: T1 12.6% med 0.389, T2 25.5% med 0.575, T3 44.4% med 0.498
- On 334 util-defined companies: days Y3 0.733 leftover after n_types 0.719 n_pos=77; util-trait Y3 0.633 leftover after days 0.622; n_types leftover after days 0.615.
- Last-month util≥0.90 companies 62. Dummy leftover after days 0.676 ρ=-0.891 fake=True.
- Leave-one-group trait leftover-after-days n=235 min/med/max 0.544 / 0.622 / 0.645.
- Util-defined overlap last-month: loc 175/334 fact 18 conf 58.
- Trait leftover after days_lag1 0.625 ρ=0.042 fake=False; after days+lag1 0.615.
- Trait leftover after last-month days+n_types invoiced — dark — (access ≠ ERP).
- Last-month util defined by first_month year: 2024 27.8%, 2025 25.6%, 2026 31.2%
- Company bootstrap leftover after last-month days+n_types n=32 p05/p50/p95 0.355 / 0.572 / 0.665.
- Trait leftover after last-month a_n_tx 0.615; after last-month days+n_types+size 0.625; after last-month days+n_types+OGTG 0.618.
- Last-month dummy leftover after days+n_types 0.530 ρ=0.383; after days+n_types+size 0.528.
- Leftover after last-month days+n_types T1/T2/T3 — / — / —.
- Last-month fac>0 dummy leftover after days 0.449 fake=False. Util-trait leftover after fac-dummy 0.633; after days+fac-dummy 0.622.
- Trait leftover-after-last-month days+n_types+has_loc rank folds 0.752 0.573 0.582 0.402 0.773 rank 0.616.
- On 334 util-defined companies leftover after last-month days+n_types 0.617 ρ=0.022 n_pos=77.
- Y2 leftover after last-month days+n_types 0.548 ρ=0.022 fake=False.
- Trait leftover after last-month days+n_types+has_loc+size 0.622 ρ=0.022.
- Trait leftover-after-last-month days+n_types+has_loc+size rank folds 0.758 0.573 0.599 0.412 0.767 rank 0.622.
- Last-month dummy leftover after days+n_types+has_loc 0.530 ρ=0.374 (dies).
- Trait leftover after last-month days+n_types+has_confirming 0.608.
- Trait leftover after last-month days+n_types+has_factoring 0.619.
- Last-month util>0 companies 298 zeros 36. Positive-util dummy leftover after days 0.473 ρ=-0.351 fake=False.
- Trait leftover after last-month days+n_types+has_any(loc/fact/conf) 0.617.
- On util>0 companies leftover after last-month days+n_types 0.627 n_pos=59.
- Trait leftover-after-last-month days+n_types+has_any rank folds 0.739 0.573 0.597 0.408 0.767 rank 0.617.
- On util>0 leftover-after-last-month days+n_types rank folds 0.789 0.606 0.677 0.457 0.604 rank 0.627 n_pos=59.
- Trait leftover after last-month days+n_types+has_any+size 0.624 ρ=0.022.
- Trait leftover after last-month days+n_types+has_any+OGTG 0.617 ρ=0.027.
- Y2 leftover after last-month days+n_types+has_any 0.548 ρ=0.023.
- Trait leftover-after-last-month days+n_types+has_any+OGTG rank folds 0.717 0.599 0.617 0.393 0.762 rank 0.617.
- Trait leftover after last-month days+n_types+has_any+OGTG+size 0.627 ρ=0.022.
- Trait leftover-after-last-month days+n_types+has_any+OGTG+size rank folds 0.733 0.603 0.643 0.388 0.766 rank 0.627.
- Y2 leftover after last-month days+n_types+has_any+OGTG+size 0.552 ρ=0.022.
- On 334 leftover after last-month days+n_types+has_any+OGTG+size 0.627 n_pos=77.
- On util>0 leftover after last-month days+n_types+has_any+OGTG+size 0.629 n_pos=59.
- Leftover after last-month days+n_types+has_any+OGTG+size invoiced — dark — (access ≠ ERP).
- Last-month dummy leftover after last-month days+n_types+has_any+OGTG+size 0.711 (above 0.55).
- On 334 leftover-after-last-month days+n_types+has_any+OGTG+size rank folds 0.733 0.603 0.643 0.388 0.766 rank 0.627 n_pos=77.
- On util>0 leftover-after-last-month days+n_types+has_any+OGTG+size rank folds 0.783 0.637 0.729 0.435 0.560 rank 0.629 n_pos=59.
- Inverse leftover of last-month OGTG after last-month util 0.642 ρ=-0.759 fake=False; util leftover after last-month OGTG 0.637; after last-month OGTG+days 0.620.
- On 334 util-defined companies last-month OGTG leftover after last-month days 0.705 n_pos=77 ρ=-0.917 fake=True (OGTG PARK leftover 0.540).
- Q6 inverse: days leftover after util0_lag1 0.711 ρ=— fake=False; days_lag1 leftover after util0_lag1 0.684 (snapshot lag cannot soak days; Q6 CLOSE).
- fillna0 hole leftover after last-month days invoiced 0.761 ρ=-0.957 fake=True dark 0.725 ρ=-0.957 fake=True (hole is last-month dummy; leftover of last-month days).
- Company bootstrap leftover after last-month days+n_types+has_any+OGTG+size n=23 p05/p50/p95 0.375 / 0.605 / 0.697.
- Leave-one-group leftover after last-month days+n_types+has_any+OGTG+size n=235 min/med/max 0.579 / 0.627 / 0.640.
- On 334 inverse last-month days leftover after last-month OGTG 0.693 ρ=0.010 fake=False; OGTG leftover after last-month days+util 0.631 ρ=-0.016 fake=False; last-month OGTG leftover after last-month days on all companies 0.718 (PARK 0.540 quote).
- Trait leftover after last-month days+n_types+has_any+OGTG+size+a_n_tx 0.626 ρ=0.023 folds 0.729 0.603 0.643 0.389 0.766; Y2 0.552.
- Permute last-month util leftover-after-penta null p50 0.479 p(obs≥null)=0.000 obs=0.627.
- OGTG PARK recipe leftover after last-month days+n_types on OGTG-defined 0.540 n_pos=402 ρ=0.291 fake=False (quote 0.540); util leftover on same book 0.617; OGTG leftover on 334 util-defined 0.531.
- Util leftover after last-month days+n_types+has_any+OGTG+size on OGTG-defined book 0.627 folds 0.733 0.603 0.643 0.388 0.766 n_pos=77; Y2 leftover after last-month days+n_types on that book 0.548.
- Inverse leftover of last-month days after last-month util+OGTG+n_types 0.652 ρ=0.046 fake=False; last-month n_types leftover after last-month util+OGTG 0.474 (days 0.711 bar should live).

Elapsed 52s.

Did **not**: overwrite `debt_schedule_qa` / `ogtg_qa.*` / `factoring_qa.*` / `n_types_qa.*` / `ar_open_qa.*` / `ap_open_qa.*`, edit `debt.py` / `gbm_core.py`, put util on the 15-col card, grow TURNOVER, invent a Y, add y10 to FROZEN_ACCEPTED, write 0–100, fit holdout, touch `product/`.
