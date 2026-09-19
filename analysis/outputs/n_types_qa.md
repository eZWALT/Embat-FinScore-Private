# Unused leftover of `f_n_types` after `c_n_days_with_tx`

Generated `2026-09-19T06:13:02+02:00` by agent `b17e9c44`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. Do not invent `y_n_types`. Do not put `f_n_types` on the 15-col card. Do not overwrite `factoring_qa.*` / `ds_r_qa.*` / `fc_r_qa.*` / `tax_month_qa.*`. Y3 never B. Night Y3 **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 TURNOVER **0.720 / 0.712**.

`f_n_types` = COUNT(DISTINCT debt type) as-of period_end. Connection inventory, rise-only. `f_has_*` already DROP from the 44. `f_n_facilities` is the cluster twin.

## Headline

DROP from the 44 as Y3 X. Leftover after days **CLOSE** rank 0.534 (OLS 0.530, fake_ols=False). Y3 n_types 0.578 vs days 0.711 vs size 0.617 vs facilities 0.579. After facilities 0.515 R²=0.358. Inverse 0.688. Twin vs facilities=True ρ=0.994. Rise-only=True drops=0. Q6 CLOSE. DROP from the 44 as Y3 X. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged. Do not put f_n_types on the 15-col card.

## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| f_n_types leftover after days | **CLOSE** | rank 0.534 OLS 0.530 fake_ols=False ρ(resid,days)=-0.360 |
| as Y3 X (not on 15-col card) | **DROP from the 44 as Y3 X** | twin=['f_n_facilities'] ρ_fac=0.994; leftover after days 0.534 dies; after facilities 0.515 R²=0.358; beat_size=-0.039. Do not put on the 15-col card. Leftover-after-days is CLOSE (dies); 44-col is DROP (facilities twin). |
| Twin / SIZE | twin=YES SIZE=no | ρ days 0.321 facilities 0.994 size 0.308 |
| Rewrite of f_n_facilities? | **YES** | leftover after facilities 0.515 R²=0.358 |
| Rise-only inventory | **YES** | f_n_types rises 344 / drops 0 / flat 19599 (rise-only). f_n_facilities rises 555 / drops 0 (factoring-qa 555/0 CONFIRM). Y3 rise-month dummy 0.508. |
| Q6 lag leftover after days_lag1 | **CLOSE** | Y3 n_types now 0.578 lag1 0.573 lag3 0.566; short raw 0.591. Days lag1 0.684 (KEEP 0.684 CONFIRM). n_types_lag1 leftover after days_lag1 0.529; short leftover 0.526. Q6 CLOSE. |
| Ever vs month | Ever n_types>0 companies 354/1214. Ever-max Y3 0.598 leftover after days 0.452. Company-mean Y3 0.600 leftover after days-mean 0.449. Demean Y3 0.449. Month leftover after ever-max 0.560 (trait vs month shock). | ever leftover 0.452 month-after-ever 0.560 |
| as health Y | **PARK** | Rise-only connection inventory, not a FICO label. Do not invent y_n_types. |
| Night quotes | **unchanged** | Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712 |


## 1. Coverage; twins; SIZE

Train f_n_types cov 100% modal0 77.1% (feature-report 77.1% CONFIRM). acf1 0.795 (0.80 CONFIRM) acf3 0.696. ρ vs log1p(a_in3) 0.308 (0.290 CONFIRM; not SIZE). mean 0.365 max 8.

| value | n | share |
| --- | --- | --- |
| 0 | 16,303 | 77.1% |
| 1 | 3,048 | 14.4% |
| 2 | 1,160 | 5.5% |
| 3 | 376 | 1.8% |
| 4 | 160 | 0.8% |
| 5 | 82 | 0.4% |
| 6 | 18 | 0.1% |
| 7 | 3 | 0.0% |
| 8 | 7 | 0.0% |


Spearman twins |ρ|≥0.80: ['f_n_facilities']. vs days 0.321 vs a_n_tx 0.318 vs f_n_facilities 0.994 (cluster twin 0.99 CONFIRM) vs loc 0.706 vs fact 0.191 vs conf 0.390 vs size 0.308.

| vs | ρ | twin |
| --- | --- | --- |
| c_n_days_with_tx | 0.321 |  |
| a_n_tx | 0.318 |  |
| f_n_facilities | 0.994 | YES |
| f_has_loc | 0.706 |  |
| f_has_factoring | 0.191 |  |
| f_has_confirming | 0.390 |  |
| f_new_facility | 0.317 |  |
| log1p(a_in3) | 0.308 |  |


## 2. Single-feature train group-fold AUROC

Y3 f_n_types 0.578 vs days 0.711 (0.711 CONFIRM) vs size 0.617 (0.617 CONFIRM, Δ -0.039) vs f_n_facilities 0.579 vs loc 0.546 (0.546 CONFIRM) vs fact 0.505 (0.505 CONFIRM) vs conf 0.485 (0.485 CONFIRM). Y2 n_types 0.523.

| y | feature | n | n_pos | CV | sign | folds |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | f_n_types | 5,648 | 402 | 0.578 | -1 | 0.638 0.574 0.630 0.504 0.543 |
| y3_recover_cash_6m | f_n_facilities | 5,648 | 402 | 0.579 | -1 | 0.637 0.576 0.627 0.508 0.546 |
| y3_recover_cash_6m | f_has_loc | 5,648 | 402 | 0.546 | -1 | 0.538 0.564 0.584 0.518 0.526 |
| y3_recover_cash_6m | f_has_factoring | 5,648 | 402 | 0.505 | -1 | 0.509 0.500 0.507 0.501 0.508 |
| y3_recover_cash_6m | f_has_confirming | 5,648 | 402 | 0.485 | -1 | 0.487 0.496 0.504 0.451 0.485 |
| y3_recover_cash_6m | f_new_facility | 5,648 | 402 | 0.515 | -1 | 0.512 0.517 0.521 0.508 0.515 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | f_n_types>0 | 5,648 | 402 | 0.580 | -1 | 0.641 0.576 0.631 0.512 0.541 |
| y2_neg_2of3 | f_n_types | 17,356 | 1,271 | 0.523 | 1 | 0.541 0.539 0.446 0.563 0.524 |
| y2_neg_2of3 | f_n_facilities | 17,356 | 1,271 | 0.521 | 1 | 0.536 0.536 0.448 0.562 0.524 |
| y2_neg_2of3 | f_has_loc | 17,356 | 1,271 | 0.523 | 1 | 0.532 0.552 0.485 0.527 0.519 |
| y2_neg_2of3 | f_has_factoring | 17,356 | 1,271 | 0.496 | 1 | 0.490 0.500 0.495 0.499 0.496 |
| y2_neg_2of3 | f_has_confirming | 17,356 | 1,271 | 0.504 | 1 | 0.511 0.497 0.480 0.505 0.529 |
| y2_neg_2of3 | f_new_facility | 17,356 | 1,271 | 0.501 | -1 | 0.510 0.471 0.508 0.511 0.505 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 1 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | log1p(a_in3) | 14,968 | 1,044 | 0.552 | 1 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | f_n_types>0 | 17,356 | 1,271 | 0.521 | 1 | 0.539 0.534 0.445 0.567 0.522 |


## 3. Honest leftover after days + inverse + facilities

Y3 leftover after days OLS 0.530 rank 0.534 ρ(resid,days)=-0.360 R²=0.089 (OLS and rank agree; resid is not a days clone). After f_n_facilities rank 0.515 R²=0.358 (weaker rewrite of n_facilities). Inverse days after n_types 0.688 (0.711 bar lives). Honest leftover after days DIES.

| y | control | OLS | rank | ρ(resid,ctrl) | R² | fake | honest_dies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | after days | 0.530 | 0.534 | -0.360 | 0.089 |  | YES |
| y3_recover_cash_6m | after f_n_facilities | 0.578 | 0.515 | 0.900 | 0.358 | YES | YES |
| y3_recover_cash_6m | after a_n_tx | 0.528 | 0.533 | -0.364 | 0.055 |  | YES |
| y3_recover_cash_6m | after size | 0.466 | 0.466 | -0.345 | 0.053 |  | YES |
| y3_recover_cash_6m | after loc | 0.532 | 0.535 | 0.052 | 0.526 |  | YES |
| y3_recover_cash_6m | after days+facilities | 0.534 | 0.590 | -0.394 | 0.396 |  | no |
| y2_neg_2of3 | after days | 0.460 | 0.461 | -0.360 | 0.089 |  | YES |
| y3_recover_cash_6m | days after n_types (inverse) | 0.694 | 0.688 | 0.074 | 0.089 |  | no |
| y3_recover_cash_6m | facilities after n_types | 0.546 | 0.513 | -0.548 | 0.358 |  | YES |


## 4. Rise-only (G/F inventory pattern)

f_n_types rises 344 / drops 0 / flat 19599 (rise-only). f_n_facilities rises 555 / drops 0 (factoring-qa 555/0 CONFIRM). Y3 rise-month dummy 0.508.

## 5. Dark vs ERP leftover

Last-month ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). mean n_types invoiced 0.361 dark 0.374. Y3 leftover invoiced 0.464 dark 0.482.

| slice | n_cm | ever>0 | mean | Y3 raw | leftover days |
| --- | --- | --- | --- | --- | --- |
| invoiced_744 | 13,554 | 212 | 0.361 | 0.593 | 0.464 |
| dark_470 | 7,603 | 142 | 0.374 | 0.588 | 0.482 |


## 6. Q6 — lag leftover after days_lag1

Y3 n_types now 0.578 lag1 0.573 lag3 0.566; short raw 0.591. Days lag1 0.684 (KEEP 0.684 CONFIRM). n_types_lag1 leftover after days_lag1 0.529; short leftover 0.526. Q6 CLOSE.

| col | n | n_pos | CV |
| --- | --- | --- | --- |
| f_n_types | 5,648 | 402 | 0.578 |
| f_n_types_lag1 | 5,648 | 402 | 0.573 |
| f_n_types_lag3 | 5,078 | 355 | 0.566 |
| c_n_days_with_tx | 5,648 | 402 | 0.711 |
| c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 |


## 7. Ever-n vs this-month count

Ever n_types>0 companies 354/1214. Ever-max Y3 0.598 leftover after days 0.452. Company-mean Y3 0.600 leftover after days-mean 0.449. Demean Y3 0.449. Month leftover after ever-max 0.560 (trait vs month shock).

f_n_types ICC 0.984 (feature-report 0.98 CONFIRM BETWEEN).

## Extra — holdout coverage

Holdout coverage only (no fit): 72 co / 1,073 CM, mean 0.390 P(>0) 20.0%.

## Extra — fold-wise leftover

Y3 leftover-after-days rank folds 0.425 0.574 0.481 0.589 0.600 spread 0.175.

| fold | OLS | rank |
| --- | --- | --- |
| 0 | 0.422 | 0.425 |
| 1 | 0.572 | 0.574 |
| 2 | 0.479 | 0.481 |
| 3 | 0.581 | 0.589 |
| 4 | 0.594 | 0.600 |


## Extra — SIZE terciles

SIZE tercile leftover after days T1/T2/T3 0.567 / 0.433 / 0.471.

| tercile | n | Y3 raw | leftover days |
| --- | --- | --- | --- |
| T1 | 6,243 | 0.483 | 0.567 |
| T2 | 6,243 | 0.556 | 0.433 |
| T3 | 6,243 | 0.585 | 0.471 |


## Extra — short / long books

Inventory leftover after days short/mid/long 0.525 / 0.507 / —.

| book | n | Y3 raw | leftover days |
| --- | --- | --- | --- |
| short_<12 | 12,474 | 0.591 | 0.525 |
| mid_12_17 | 4,740 | 0.558 | 0.507 |
| long_>=18 | 3,943 | LOW_POWER | — |


## Extra — Y3 rate by n_types

Y3 rate by n_types bucket: 0 8.8%, 1 3.8%, 2 2.8%, 3+ 5.6%

| n_types | n | Y3+ | rate |
| --- | --- | --- | --- |
| 0 | 3,772 | 331 | 8.8% |
| 1 | 1,141 | 43 | 3.8% |
| 2 | 469 | 13 | 2.8% |
| 3+ | 266 | 15 | 5.6% |


## Extra — debt product type mix

debt_products types 8: loan 1022, lineofcredit 536, confirming 229, leasing 179, guarantee 155, mortgage 60, renting 34, factoring 24. n_types max 8 is the type inventory, not facilities.

| type | n | share |
| --- | --- | --- |
| loan | 1,022 | 45.6% |
| lineofcredit | 536 | 23.9% |
| confirming | 229 | 10.2% |
| leasing | 179 | 8.0% |
| guarantee | 155 | 6.9% |
| mortgage | 60 | 2.7% |
| renting | 34 | 1.5% |
| factoring | 24 | 1.1% |


## Extra — leftover on f_has_* = 1 months

Leftover after days on flag=1 months loc — fact — conf —.

| flag | n | Y3 raw | leftover days | n_pos |
| --- | --- | --- | --- | --- |
| f_has_loc | 2,482 | LOW_POWER | — | 36 |
| f_has_factoring | 151 | LOW_POWER | — | 0 |
| f_has_confirming | 692 | LOW_POWER | — | 17 |


Plot: `n_types_qa.png`.

## What failed / next

- no replica miss; leftover after days CLOSE 0.534 (dies) and twin of f_n_facilities ρ 0.994 → DROP from the 44. T1 leftover sometimes lives (boot p50 0.579) but raw loses to size and the twin gate still kills KEEP.

## Later extras (same module)

- Company bootstrap leftover-after-days n=80 p05/p50/p95 0.425 / 0.533 / 0.573 P(≥0.55)=37.5%.
- Pearson n_types~n_facilities 0.598 (Spearman 0.994). Equal on 87.0% of CM; disagree 2,744 CM / 213 companies. P(>0) agree 100.0%. Y3 leftover on disagree CM — raw —.
- SIZE tercile leftover after days T1/T2/T3 0.567 / 0.433 / 0.471.
- Permute n_types leftover-after-days null p50 0.631 p(obs≥null)=1.000 obs=0.534.
- Leave-one-group leftover-after-days n=235 min/med/max 0.479 / 0.534 / 0.543.
- Inventory leftover after days short/mid/long 0.525 / 0.507 / —.
- Twin f_n_facilities leftover after days 0.534 (n_types was 0.534). n_types leftover after has_* flags 0.579. binary >0 leftover after days 0.536. Fold-wise OOF leftover after days 0.534 after facilities 0.514. On n_types>0 months leftover after days 0.691.
- Rise-month leftover after days 0.695 ρ(resid,days)=-0.943 fake=True n_pos=402; first-rise leftover 0.695 ρ(resid,days)=-0.943 fake=True.
- On n_types>0 CM (4,854): Y3 raw 0.506 n_pos=71 days-in-slice 0.732. Leftover after days 0.691 ρ(resid,days)=-0.360 fake=False. After facilities 0.521 R²=0.358. After both 0.566.
- Company-mean shuffle leftover-after-days null p50 0.587 p(obs≥null)=1.000 obs=0.534. Pure N(0,1) leftover after days 0.504 ρ(resid,days)=-0.003.
- Y3 rate by n_types bucket: 0 8.8%, 1 3.8%, 2 2.8%, 3+ 5.6%
- Months-since-first-type leftover after days 0.477 raw 0.506 ρ(resid,days)=0.005 n_pos=71.
- Bootstrap leftover after f_n_facilities n=24 p05/p50/p95 0.453 / 0.518 / 0.545.
- Leftover after facilities invoiced 0.468 dark 0.545.
- Ever n_types>0 companies 354/1214 recover 33 (13.3%); never 860 recover 141 (29.6%). Leftover of n_types after has_loc+fact+conf 0.532. Δ=n_facilities-n_types Y3 0.546 leftover after days 0.607. Largest group among connected 3.7%.
- Δ=n_facilities−n_types Y3 0.546 leftover after days 0.607 ρ(resid,days)=-0.650 fake=False. After n_types 0.513. ρ size 0.273 days 0.278 fac 0.790. beat_size=-0.070. KEEP-as-X of Δ=no (raw loses to size; leftover 0.607 is not an X).
- debt_products types 8: loan 1022, lineofcredit 536, confirming 229, leasing 179, guarantee 155, mortgage 60, renting 34, factoring 24. n_types max 8 is the type inventory, not facilities.
- Inventory cadence companies always0=860 mixed-rise=277 always>0=89. Mixed leftover after days 0.471 n_pos=56. Always>0 leftover — n_pos=32.
- Leftover after days+size 0.530 ρ(resid,days)=-0.314. After days+size+facilities 0.578.
- Banking cousin g_n_types Y3 0.552 ρ vs f_n_types 0.395 not a twin. f_n_types leftover after g_n_types 0.570; after days+g 0.442.
- Other-type-only companies (n_types>0, no loc/fact/conf) 148. Y3 raw — leftover after days — n_pos=44 ever-recover 14.6%.
- Last-6m leftover after days — n_pos=0. First-6m leftover after days 0.470 n_pos=118.
- T1 bootstrap leftover-after-days n=20 p05/p50/p95 0.419 / 0.579 / 0.651.
- Month other-type (n_types>0, no loc/fact/conf) share 9.9% Y3 0.534 leftover after days 0.637 ρ(resid,days)=-0.753 fake=False.
- Other-type leftover after days+size 0.639 ρ(resid,days)=-0.711; after facilities 0.513; after n_types 0.511 (0.637 after days is near-days leak ρ=-0.753).
- Dark∩T1 leftover after days 0.521 raw 0.460 n_pos=52.
- Invoiced∩T1 leftover after days 0.512 raw 0.560 n_pos=118.
- Ever-connected leftover after days 0.619 raw 0.388 n_pos=88. Ever n_types≥2 leftover — raw — n_pos=29.
- Ever-connected leftover after days 0.619 ρ(resid,days)=-0.360 fake=False; after facilities 0.534; after both 0.573.
- Leftover after days on flag=1 months loc — fact — conf —.
- Loan-book companies 230. Y3 n_types 0.484 leftover after days 0.607 after facilities 0.385 n_pos=56.
- Leasing/guarantee/mortgage/renting companies 92. Y3 n_types — leftover after days — after facilities — n_pos=25.
- Ever-connected leftover after days dark — n_pos=38 invoiced 0.615 n_pos=50. After facilities dark — invoiced 0.545.
- Rank residual of n_types after days ICC 0.984 (BETWEEN leftover).
- Last Y3-defined month leftover after days 0.471 raw 0.575 n=725 n_pos=154 (company-level leftover).
- First Y3-defined month leftover after days 0.482 raw 0.556 n=725 n_pos=91.
- Last Y3 month among n_types>0 leftover after days — raw — after facilities — n_pos=28.
- Ever-connected bootstrap leftover-after-days n=24 p05/p50/p95 0.358 / 0.576 / 0.639.

Elapsed 39s.

Did **not**: overwrite `factoring_qa.*` / `ds_r_qa.*` / `fc_r_qa.*` / `tax_month_qa.*` / `n_cust_qa.*` / `issued_qa.*` / `in3_qa.*`, edit `debt.py` / `gbm_core.py`, put n_types on the 15-col card, grow TURNOVER, invent `y_n_types`, write 0–100, fit holdout, touch `product/`.
