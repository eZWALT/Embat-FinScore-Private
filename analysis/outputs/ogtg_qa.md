# Unused leftover of `f_outstanding_gt_granted` after `c_n_days_with_tx`

Generated `2026-09-19T06:42:08+02:00` by agent `b17e9c44`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. Do not invent a Y. Do not put OGTG on the 15-col card. Do not overwrite `factoring_qa.*` / `n_types_qa.*` / `ds_r_qa.*` / `fc_r_qa.*` / `debt_schedule_qa` / `sib_neg_qa.*` / `ap_issued_qa.*`. Y3 never B. Night Y3 **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 TURNOVER **0.720 / 0.712**.

`f_outstanding_gt_granted` = clean snapshot flag, max over as-of facilities. NaN until the 2026-08 extract. NORTH_STAR PARK snapshot cols as X. Same object as `y10_ogtg_last_month` PARK.

## Headline

PARK as snapshot X. Leftover after days **CLOSE** native n_pos=0 hole-rank 0.711 OLS 0.711. Last-month-only=True cov 5.7% vs util 1.6%. Y3 native — vs days 0.711 vs size 0.617 vs n_types 0.578. Twin=none ρ days 0.085 types 0.278 util 0.152. Rise/extract True n1=32. Q6 CLOSE. PARK as Y / DROP from the 44 as Y3 X. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged. Do not put OGTG on the 15-col card.

## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| OGTG leftover after days | **CLOSE** | native n_pos=0 hole-rank 0.711 OLS 0.711 fake=True |
| as Y3 X (not on 15-col card) | **PARK as Y / DROP from the 44 as Y3 X** | Last-month extract flag (cov 5.7%; Y3 n_pos on snapshot=0). Snapshot cannot lead (Q6 CLOSE). NORTH_STAR PARK snapshot cols as X. Hole leftover after days 0.711. Do not put on the 15-col card. Same object as y10_ogtg_last_month PARK. |
| Last-month-only / snapshot | **YES** | Train OGTG defined 1,214/21,157 (5.7%; quote 5.7% CONFIRM). f_util_snapshot 1.6% (CONFIRM 1.6%). Defined periods ['2026-08'] last-month-only=True (last 2026-08 n1=32 n0=1182; quote 32 CONFIRM). modal0-as-zero 97.4% (97.4% CONFIRM). ρ vs log1p(a_in3) on defined 0.102 (not SIZE). acf1 —. |
| Twin / SIZE | twin=no SIZE=no | ρ days 0.085 types 0.278 util 0.152 size 0.102 |
| Q6 lag leftover | **CLOSE** | Snapshot cannot lead: native OGTG Y3 n_pos=0 (LOW_POWER on extract month). fillna0 lag1 leftover after days_lag1 0.632. Days lag1 0.684 (KEEP 0.684 CONFIRM). Q6 CLOSE. |
| Rise / extract hole | **extract-hole** | OGTG rises 0 / drops 0 / defined diffs 0 (extract-hole: no within-company pair). Last-month =1 companies 32 (quote 32 CONFIRM). |
| as health Y | **PARK** | Same as y10_ogtg_last_month. Snapshot extract, not a FICO label. |
| Night quotes | **unchanged** | Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712 |


## 1. Coverage — last-month vs growing panel

Train OGTG defined 1,214/21,157 (5.7%; quote 5.7% CONFIRM). f_util_snapshot 1.6% (CONFIRM 1.6%). Defined periods ['2026-08'] last-month-only=True (last 2026-08 n1=32 n0=1182; quote 32 CONFIRM). modal0-as-zero 97.4% (97.4% CONFIRM). ρ vs log1p(a_in3) on defined 0.102 (not SIZE). acf1 —.

Spearman twins |ρ|≥0.80 on defined OGTG: none. vs days 0.085 vs a_n_tx 0.090 vs n_types 0.278 vs n_facilities 0.278 vs util 0.152 vs size 0.102.

| vs | ρ | twin |
| --- | --- | --- |
| c_n_days_with_tx | 0.085 |  |
| a_n_tx | 0.090 |  |
| f_n_types | 0.278 |  |
| f_n_facilities | 0.278 |  |
| f_util_snapshot | 0.152 |  |
| log1p(a_in3) | 0.102 |  |
| f_has_loc | 0.358 |  |


## 2. Single-feature train group-fold AUROC

Y3 native OGTG — n=0 n_pos=0 (LOW_POWER — snapshot month has no Y3). fillna0 hole 0.500. vs days 0.711 (0.711 CONFIRM) vs size 0.617 (0.617 CONFIRM) vs n_types 0.578 (0.578 CONFIRM) vs util —. Y2 native —.

| y | feature | n | n_pos | CV | sign | folds |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | f_outstanding_gt_granted | 0 | 0 | LOW_POWER | — | — |
| y3_recover_cash_6m | ogtg_fillna0 | 5,648 | 402 | 0.500 | 1 | 0.500 0.500 0.500 0.500 0.500 |
| y3_recover_cash_6m | f_util_snapshot | 0 | 0 | LOW_POWER | — | — |
| y3_recover_cash_6m | f_n_types | 5,648 | 402 | 0.578 | -1 | 0.638 0.574 0.630 0.504 0.543 |
| y3_recover_cash_6m | f_n_facilities | 5,648 | 402 | 0.579 | -1 | 0.637 0.576 0.627 0.508 0.546 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y2_neg_2of3 | f_outstanding_gt_granted | 0 | 0 | LOW_POWER | — | — |
| y2_neg_2of3 | ogtg_fillna0 | 17,356 | 1,271 | 0.500 | 1 | 0.500 0.500 0.500 0.500 0.500 |
| y2_neg_2of3 | f_util_snapshot | 0 | 0 | LOW_POWER | — | — |
| y2_neg_2of3 | f_n_types | 17,356 | 1,271 | 0.523 | 1 | 0.541 0.539 0.446 0.563 0.524 |
| y2_neg_2of3 | f_n_facilities | 17,356 | 1,271 | 0.521 | 1 | 0.536 0.536 0.448 0.562 0.524 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | log1p(a_in3) | 14,968 | 1,044 | 0.552 | 1 | 0.523 0.609 0.595 0.520 0.513 |


## 3. Honest leftover after days + inverse + n_types

Native leftover after days rank — n_pos=0 (LOW_POWER). fillna0 hole leftover after days rank 0.711 OLS 0.711 ρ(resid,days)=-0.996 fake=True. After n_types hole 0.578 after facilities hole 0.579. Inverse days after hole 0.711. Honest leftover DIES.

| control | OLS | rank | n | n_pos | ρ(resid,ctrl) | fake | dies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native after days | — | — | 0 | 0 | -0.921 | YES | YES |
| native after n_types | — | — | 0 | 0 | -0.803 | YES | YES |
| native after n_facilities | — | — | 0 | 0 | -0.804 | YES | YES |
| hole after days | 0.711 | 0.711 | 5648 | 402 | -0.996 | YES | YES |
| hole after n_types | 0.578 | 0.578 | 5648 | 402 | -0.986 | YES | YES |
| hole after n_facilities | 0.579 | 0.579 | 5648 | 402 | -0.986 | YES | YES |
| hole after size | 0.617 | 0.617 | 5528 | 391 | -0.994 | YES | YES |
| days after native OGTG (inverse) | — | — | 0 | 0 | 0.011 |  | YES |
| days after hole fillna0 (inverse) | 0.711 | 0.711 | 5648 | 402 | -0.000 |  | no |


## 4. Rise-only / extract-hole

OGTG rises 0 / drops 0 / defined diffs 0 (extract-hole: no within-company pair). Last-month =1 companies 32 (quote 32 CONFIRM).

## 5. Dark vs ERP leftover

Last-month ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). ogtg=1 invoiced 24 dark 8. Hole leftover after days invoiced 0.730 dark 0.704.

| slice | n | defined | ogtg=1 | rate |
| --- | --- | --- | --- | --- |
| invoiced_744 | 744 | 744 | 24 | 3.2% |
| dark_470 | 470 | 470 | 8 | 1.7% |


## 6. Q6 — snapshot cannot lead

Snapshot cannot lead: native OGTG Y3 n_pos=0 (LOW_POWER on extract month). fillna0 lag1 leftover after days_lag1 0.632. Days lag1 0.684 (KEEP 0.684 CONFIRM). Q6 CLOSE.

| col | n | n_pos | CV |
| --- | --- | --- | --- |
| f_outstanding_gt_granted | 0 | 0 | LOW_POWER |
| ogtg0 | 5,648 | 402 | 0.500 |
| ogtg0_lag1 | 5,648 | 402 | 0.500 |
| c_n_days_with_tx | 5,648 | 402 | 0.711 |
| c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 |


## 7. Ever / last-month trait

Company last-month OGTG=1 32/1214. Trait leftover after days 0.684 raw 0.518 n_pos=402. Ever-Y3 recover ogtg 3/32 (12.5%) vs rest 24.4% (debt-schedule 3/24=12.5% / 24.4% on labeled).

Last-month labeled Y3 0 Y2 0 (debt-schedule horizon → almost empty; quote Y3=0). OGTG defined 1214 =1 32.

## Extra — holdout coverage

Holdout coverage only (no fit): 72 co / 1,073 CM, defined 72 last-month =1 3.

## Extra — SIZE terciles of trait leftover

Trait leftover after days T1/T2/T3 0.598 / 0.612 / 0.696 (all fake days leaks if ρ≥0.80).

| tercile | leftover | ρ(resid,days) | fake |
| --- | --- | --- | --- |
| T1 | 0.598 | -0.915 | True |
| T2 | 0.612 | -0.915 | True |
| T3 | 0.696 | -0.915 | True |


## Extra — last-month OGTG=1 by SIZE tercile

Last-month OGTG=1 by SIZE tercile: T1 1.5%, T2 1.2%, T3 5.2%

| tercile | n | ogtg=1 | rate |
| --- | --- | --- | --- |
| T1 | 405 | 6 | 1.5% |
| T2 | 404 | 5 | 1.2% |
| T3 | 405 | 21 | 5.2% |


## Extra — OGTG=1 by first_month year

Last-month OGTG=1 by first_month year: 2024 3.6%, 2025 1.5%, 2026 2.9%

| first_year | n | ogtg=1 | rate |
| --- | --- | --- | --- |
| 2024 | 525 | 19 | 3.6% |
| 2025 | 481 | 7 | 1.5% |
| 2026 | 208 | 6 | 2.9% |


## Extra — coverage by month (growing panel vs extract)

Month table: 1/24 months have any OGTG defined (growing panel would be 24; snapshot is one extract month).

| period | n | ogtg_def | ogtg=1 | util_def | y3_pos |
| --- | --- | --- | --- | --- | --- |
| 2024-09 | 435 | 0 | 0 | 0 | 0 |
| 2024-10 | 473 | 0 | 0 | 0 | 2 |
| 2024-11 | 483 | 0 | 0 | 0 | 11 |
| 2024-12 | 525 | 0 | 0 | 0 | 9 |
| 2025-01 | 636 | 0 | 0 | 0 | 9 |
| 2025-02 | 677 | 0 | 0 | 0 | 14 |
| 2025-03 | 714 | 0 | 0 | 0 | 22 |
| 2025-04 | 733 | 0 | 0 | 0 | 20 |
| 2025-05 | 752 | 0 | 0 | 0 | 22 |
| 2025-06 | 762 | 0 | 0 | 0 | 28 |
| 2025-07 | 796 | 0 | 0 | 0 | 37 |
| 2025-08 | 833 | 0 | 0 | 0 | 30 |
| 2025-09 | 864 | 0 | 0 | 0 | 24 |
| 2025-10 | 908 | 0 | 0 | 0 | 31 |
| 2025-11 | 945 | 0 | 0 | 0 | 33 |
| 2025-12 | 1,006 | 0 | 0 | 0 | 35 |
| 2026-01 | 1,132 | 0 | 0 | 0 | 36 |
| 2026-02 | 1,204 | 0 | 0 | 0 | 39 |
| 2026-03 | 1,211 | 0 | 0 | 0 | 0 |
| 2026-04 | 1,212 | 0 | 0 | 0 | 0 |
| 2026-05 | 1,214 | 0 | 0 | 0 | 0 |
| 2026-06 | 1,214 | 0 | 0 | 0 | 0 |
| 2026-07 | 1,214 | 0 | 0 | 0 | 0 |
| 2026-08 | 1,214 | 1214 | 32 | 334 | 0 |


## Extra — company last-month Spearman (snapshot twins)

Company last-month Spearman OGTG vs days 0.085 n_types 0.278 facilities 0.278 util 0.152 size 0.102 (no last-month twin |ρ|≥0.80).

| vs | ρ | twin |
| --- | --- | --- |
| last-month days | 0.085 |  |
| last-month a_n_tx | 0.090 |  |
| last-month n_types | 0.278 |  |
| last-month n_facilities | 0.278 |  |
| last-month util | 0.152 |  |
| last-month size | 0.102 |  |


Plot: `ogtg_qa.png`.

## What failed / next

- no replica miss; native leftover undefined (Y3 n_pos=0 on 2026-08). fillna0 / last-month dummy leftover 0.711 is a fake days leak (ρ=-0.996). Trait leftover 0.684 is also a fake days leak (ρ=-0.915). Honest leftover after last-month days+n_types 0.540 dies; after LOC dummy 0.535 dies. PARK snapshot X / DROP from the 44.

## Later extras (same module)

- Last-month dummy leftover after days 0.711 ρ(resid,days)=0.881 fake=True raw 0.500. After size 0.617. Same object as fillna0 hole leftover 0.711.
- Trait leftover after days 0.684 ρ(resid,days)=-0.915 fake=True; after n_types 0.545; after facilities 0.546; after days+size 0.661.
- Company bootstrap trait leftover-after-days n=64 p05/p50/p95 0.656 / 0.688 / 0.718.
- Last-month util defined 334 OGTG=1 32 both 28. ρ OGTG~util 0.152. Hole leftover after util-fillna0 0.500 ρctrl=-0.806.
- OGTG ICC on defined 1.000 (feature-report 1.00 BETWEEN).
- Permute last-month OGTG leftover-after-days null p50 0.700 p(obs≥null)=0.917 obs=0.684.
- Trait leftover after days invoiced 0.696 ρ=-0.915 dark 0.685 ρ=-0.915.
- Q6 fillna0 lag1 leftover after days_lag1 0.632 ρ(resid,days_lag1)=— fake=False .
- OGTG=1 companies 32 in 25 groups; largest group share 12.5% (not a group dummy).
- Trait leftover after days T1/T2/T3 0.598 / 0.612 / 0.696 (all fake days leaks if ρ≥0.80).
- Company-level last-month OGTG vs ever-Y3 AUROC + 0.490 − 0.510 n=725 n_pos=174 (32-company extract dummy).
- Leave-one-group trait leftover-after-days n=235 min/med/max 0.676 / 0.684 / 0.694.
- Inverse: n_types leftover after OGTG-trait 0.574; days leftover after OGTG-trait 0.706 (0.711 bar should live).
- Last-month facilities>0 354 OGTG=1 among them 32 (9.0%). OGTG is a rare extract flag on the connected book, not a panel X.
- Y2 trait leftover after days 0.571 ρ(resid,days)=-0.915 fake=True raw 0.483.
- On last-month-connected companies leftover after days 0.694 ρ=-0.915 fake=True; after n_types 0.459.
- On the 32 OGTG=1 companies: days Y3 — leftover after n_types — n_pos=3; n_types Y3 — leftover after days —.
- Last-month dummy leftover after days+n_types 0.542 ρ(resid,days)=0.479; after days+size+types 0.586.
- Bootstrap last-month-dummy leftover-after-days n=48 p05/p50/p95 0.676 / 0.716 / 0.746.
- Last-month util vs ever-Y3 AUROC + 0.419 − 0.581 n=232 n_pos=28. ρ util~OGTG 0.152.
- Last-month dummy ICC 0.185; rank residual after days ICC 0.191.
- Last-month OGTG=1 by SIZE tercile: T1 1.5%, T2 1.2%, T3 5.2%
- OGTG=1 overlap last-month: loc 30/32 fact 4 conf 9.
- T3 last-month dummy leftover after days 0.726 ρ(resid,days)=0.881 fake=True.
- On last-month LOC companies leftover of OGTG-trait after days — ρ=-0.915 fake=True. OGTG leftover after has_loc 0.515.
- BETWEEN leftover of OGTG-trait after days-mean 0.703 ρ(resid,days-mean)=-0.911 fake=True.
- Trait leftover-after-days rank folds 0.635 0.692 0.697 0.663 0.734 (fake days leak).
- Trait leftover after a_n_tx 0.677 ρ=-0.910 fake=True. Last-month dummy leftover after a_n_tx 0.703 fake=True.
- Trait leftover after days_lag1 0.659 ρ=-0.915 fake=True; after days+lag1 0.675. Last-month dummy after days+lag1 0.537.
- Last-month tenure months OGTG=1 median 24.0 vs rest 20.0. Last-month days median OGTG=1 20.5 vs rest 13.0 (32-company extract, not a longer trail).
- Trait leftover after last-month-days-as-trait 0.718 ρ=-0.917 fake=True (BETWEEN days of the extract month, not the panel path).
- Q6 fillna0 lag3 leftover after days_lag3 0.632 ρ=— fake=False. Days lag3 0.666.
- Company last-month Spearman OGTG vs days 0.085 n_types 0.278 facilities 0.278 util 0.152 size 0.102 (no last-month twin |ρ|≥0.80).
- Trait leftover after days+n_types 0.653 ρ(resid,days)=-0.766; after days+facilities 0.651; after n_types+facilities 0.545 (above 0.55 / fake).
- Holdout OGTG defined periods ['2026-08'] last-month-only=True n1=3 (same extract hole as train).
- Last-month n_types median OGTG=1 2.0 vs rest 0.0; n_facilities median OGTG=1 4.0 vs rest 0.0 (connected book, still a 32-row extract dummy).
- Last-month dummy leftover after days_lag1 0.684 ρ=-0.865 fake=True; after days_lag3 0.666.
- Trait leftover after n_types invoiced 0.552 dark 0.567 (access ≠ ERP; leftover after inventory still dies if <0.55).
- Trait leftover after days+n_types+size 0.613 ρ=-0.410; after days+n_types+has_loc 0.653.
- Trait leftover after last-month-n_types-as-trait 0.566 ρ=-0.795; after last-month-n_types+panel-days 0.654.
- Random 32-company dummy leftover-after-days null p50 0.693 p(obs≥null)=0.812 obs=0.684 (OGTG dummy is not above a random 32).
- Y2 trait leftover after n_types 0.513; after days+n_types 0.560.
- Last-month-connected dummy leftover after days 0.449 fake=False. OGTG-trait leftover after connected dummy 0.562; after days+connected 0.663.
- Company bootstrap trait leftover-after-n_types n=32 p05/p50/p95 0.487 / 0.548 / 0.577.
- Month table: 1/24 months have any OGTG defined (growing panel would be 24; snapshot is one extract month).
- Trait leftover after days+n_types invoiced 0.668 dark 0.671 (access ≠ ERP).
- Last-month OGTG=1 by first_month year: 2024 3.6%, 2025 1.5%, 2026 2.9%
- Inverse: connected-dummy leftover after OGTG-trait 0.591. Last-month dummy leftover after connected dummy 0.594.
- Q6 fillna0 lag1 leftover after contemporaneous days 0.500 ρ=— fake=False; lag3 after days 0.500 (snapshot lag cannot lead).
- Trait leftover after days by first_year 2024/2025/2026 0.699 / 0.666 / —.
- Last-month-LOC dummy leftover after days 0.566. OGTG leftover after LOC dummy 0.535; after days+LOC 0.596.
- OGTG leftover after connected dummy invoiced 0.578 dark 0.553 (access ≠ ERP).
- Last-month has_any(loc/fact/conf) dummy leftover after days 0.561. OGTG leftover after has_any 0.537; after days+has_any+n_types 0.532.
- Inverse: LOC dummy leftover after OGTG-trait 0.561. Days leftover after OGTG+LOC 0.692 (0.711 bar should live).
- Trait leftover-after-n_types rank folds 0.599 0.510 0.624 0.455 0.539 rank 0.545 (dies).
- OGTG leftover after has_any invoiced 0.541 dark 0.549 (access ≠ ERP).
- Last-month-util-defined dummy leftover after days 0.458 fake=False. OGTG leftover after util-defined dummy 0.567.
- BETWEEN leftover of OGTG-trait after n_types-mean 0.566 ρ=-0.790; after n_types-mean+days 0.459.
- Trait leftover-after-days+n_types rank folds 0.659 0.655 0.733 0.541 0.676 rank 0.653 ρ(resid,days)=-0.766.
- Inverse: n_types-mean leftover after OGTG-trait 0.593. Days leftover after OGTG+n_types-mean 0.683 (0.711 bar should live).
- Trait leftover after last-month days+n_types+has_loc 0.538 ρ=0.283 (dies).
- Trait leftover after last-month days+n_types 0.540 ρ=0.291 (dies).
- Trait leftover after last-month days+n_types+has_loc invoiced 0.531 dark 0.541 (access ≠ ERP).
- Trait leftover after last-month a_n_tx 0.730 ρ=-0.914; after last-month days+a_n_tx 0.731 (above 0.55).
- Trait leftover after last-month n_facilities 0.560; after panel-days+last-month facilities 0.644.
- Trait leftover after last-month days+n_facilities 0.542; after last-month days+n_types+size 0.611 (dies).
- Company bootstrap leftover after last-month days+n_types n=32 p05/p50/p95 0.461 / 0.574 / 0.694.
- Trait leftover after last-month days+n_types invoiced 0.532 dark 0.538 (access ≠ ERP).
- Trait leftover-after-last-month days+n_types rank folds 0.513 0.545 0.445 0.594 0.601 rank 0.540 (dies).
- Trait leftover-after-last-month days+n_types+has_loc rank folds 0.514 0.545 0.439 0.594 0.600 rank 0.538 (dies).
- Leftover after last-month days+n_types T1/T2/T3 0.606 / 0.487 / 0.454 (honest leftover dies on T3=0.454).

Elapsed 51s.

Did **not**: overwrite `factoring_qa.*` / `n_types_qa.*` / `ds_r_qa.*` / `fc_r_qa.*` / `debt_schedule_qa` / `sib_neg_qa.*` / `ap_issued_qa.*`, edit `debt.py` / `gbm_core.py`, put OGTG on the 15-col card, grow TURNOVER, invent a Y, write 0–100, fit holdout, touch `product/`.
