# Unused leftover of `c_tax_month` after `c_n_days_with_tx`

Generated `2026-09-19T05:46:40+02:00` by agent `a91c4e02`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_tax`. Do not put tax on the 15-col card. Do not overwrite `tax_qa.py` / `tax_qa.md` (missed-tax). Y3 never B. Night Y3 **0.762 / 0.752**. Days **0.711**. Size **0.617**. Salary **0.671**. SS leftover **0.635**. Y7 TURNOVER **0.720 / 0.712**.

`c_tax_month` = 1 if any category = tax this month (fillna 0). `c_missed_tax` already CLOSE as Y3 X (0.511) / PARK as Y (Q-peaked calendar dummy).

## Headline

CLOSE leftover-after-days rank 0.520 (OLS 0.520, fake_ols=False). Y3 tax 0.611 vs days 0.711 vs size 0.617 vs salary 0.671 vs ss 0.693 vs missed 0.511. After ss 0.528 after salary 0.538 stack 0.489. Inverse 0.682. Calendar q_peaked Q-gap 25.7%. Q6 CLOSE. CLOSE unused leftover. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged. Do not put tax on the 15-col card.

## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| c_tax_month leftover after days | **CLOSE** | rank 0.520 OLS 0.520 fake_ols=False ρ(resid,days)=-0.075 |
| as Y3 X (not on 15-col card) | **CLOSE unused leftover** | Y3 leftover after days rank 0.520 OLS 0.520 dies; beat_size=-0.006 twin=False. After ss 0.528 after salary 0.538 stack 0.489. Calendar q_peaked. Do not put tax on the 15-col card. |
| Twin / SIZE | twin=no SIZE=no | ρ days 0.428 ss 0.330 salary 0.317 missed -0.312 size 0.357 |
| Complement of missed-tax? | **no** | ρ missed -0.312 Jaccard 0.000 P(missed|not tax) 17.3% |
| Calendar | **q_peaked** | Tax share Q-months 69.2% vs other 43.4% (gap 25.7%, ratio 1.59; tax_qa 69.2%/43.4% CONFIRM). Shape **q_peaked** (missed-tax Q-gap 25.9% LIKE missed-tax; salary Q-gap 0.6% not salary-monthly). Leftover after is_q 0.624; leftover after days on Q-months 0.547 on other 0.501. |
| Q6 lag leftover after days_lag1 | **CLOSE** | Y3 tax now 0.611 lag1 0.597 lag3 0.597. Days lag1 0.684 (KEEP 0.684 CONFIRM). tax_lag1 leftover after days_lag1 0.519 lag3 0.518. Q6 CLOSE. |
| Inverse: days leftover after tax | **lives** | rank 0.682 OLS 0.684 |
| Night quotes | **unchanged** | Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712 |


## 1. Coverage; twins; SIZE

Train c_tax_month cov 100% P(1) 51.8% modal 51.8% (feature-report 51.8% CONFIRM). acf1 -0.125 (quote −0.13 CONFIRM) acf3 0.277. ρ vs log1p(a_in3) 0.357 (0.378 CONFIRM; not SIZE).

Spearman twins |ρ|≥0.80: none. vs days 0.428 vs a_n_tx 0.429 vs salary 0.317 vs ss 0.330 vs missed_tax -0.312 vs size 0.357. Jaccard(tax, missed) 0.000 both 0. P(missed|not tax) 17.3% — presence is NOT the complement of missed.

| vs | ρ | twin |
| --- | --- | --- |
| c_n_days_with_tx | 0.428 |  |
| a_n_tx | 0.429 |  |
| c_salary_month | 0.317 |  |
| c_ss_month | 0.330 |  |
| c_missed_tax | -0.312 |  |
| log1p(a_in3) | 0.357 |  |
| 1-c_tax_month | -1.000 |  |


## 2. Single-feature train group-fold AUROC

Y3 c_tax_month 0.611 (ss_qa 0.611 CONFIRM) vs days 0.711 (0.711 CONFIRM) vs size 0.617 (0.617 CONFIRM, Δ -0.006) vs salary 0.671 (0.671 CONFIRM) vs ss 0.693 (0.693 CONFIRM) vs missed_tax 0.511 (0.511 CONFIRM). Y2 tax 0.535.

| y | feature | n | n_pos | CV | sign | folds |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | c_tax_month | 5,648 | 402 | 0.611 | -1 | 0.556 0.624 0.630 0.613 0.630 |
| y3_recover_cash_6m | c_missed_tax | 5,648 | 402 | 0.511 | 1 | 0.532 0.504 0.487 0.517 0.515 |
| y3_recover_cash_6m | c_salary_month | 5,648 | 402 | 0.671 | -1 | 0.611 0.659 0.745 0.689 0.652 |
| y3_recover_cash_6m | c_ss_month | 5,648 | 402 | 0.693 | -1 | 0.673 0.667 0.745 0.687 0.694 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | is_q_month | 5,648 | 402 | 0.495 | 1 | 0.514 0.487 0.494 0.498 0.483 |
| y3_recover_cash_6m | 1-c_tax_month | 5,648 | 402 | 0.611 | 1 | 0.556 0.624 0.630 0.613 0.630 |
| y2_neg_2of3 | c_tax_month | 17,356 | 1,271 | 0.535 | 1 | 0.559 0.558 0.540 0.520 0.498 |
| y2_neg_2of3 | c_missed_tax | 17,356 | 1,271 | 0.514 | -1 | 0.505 0.502 0.525 0.504 0.534 |
| y2_neg_2of3 | c_salary_month | 17,356 | 1,271 | 0.522 | 1 | 0.511 0.572 0.535 0.432 0.559 |
| y2_neg_2of3 | c_ss_month | 17,356 | 1,271 | 0.539 | -1 | 0.509 0.562 0.516 0.604 0.505 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 1 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | log1p(a_in3) | 14,968 | 1,044 | 0.552 | 1 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | is_q_month | 17,356 | 1,271 | 0.502 | -1 | 0.506 0.498 0.496 0.502 0.506 |
| y2_neg_2of3 | 1-c_tax_month | 17,356 | 1,271 | 0.535 | -1 | 0.559 0.558 0.540 0.520 0.498 |


## 3. Honest leftover after days + inverse + cousins

Y3 leftover after days OLS 0.520 rank 0.520 ρ(resid,days)=-0.075 R²=0.178 (OLS and rank agree; resid is not a days clone). After missed 0.617 after salary 0.538 after ss 0.528 after days+salary+ss 0.489. Inverse days after tax 0.682 (0.711 bar lives). Honest leftover after days DIES.

| y | control | OLS | rank | ρ(resid,ctrl) | R² | fake | honest_dies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | after days | 0.520 | 0.520 | -0.075 | 0.178 |  | YES |
| y3_recover_cash_6m | after missed_tax | 0.617 | 0.617 | -0.069 | 0.098 |  | no |
| y3_recover_cash_6m | after salary | 0.538 | 0.538 | -0.114 | 0.101 |  | YES |
| y3_recover_cash_6m | after ss | 0.528 | 0.528 | -0.104 | 0.109 |  | YES |
| y3_recover_cash_6m | after size | 0.567 | 0.567 | -0.154 | 0.122 |  | no |
| y3_recover_cash_6m | after a_n_tx | 0.526 | 0.526 | -0.078 | 0.056 |  | YES |
| y3_recover_cash_6m | after days+salary+ss | 0.488 | 0.489 | -0.031 | 0.212 |  | YES |
| y3_recover_cash_6m | after is_q_month | 0.624 | 0.624 | -0.185 | 0.060 |  | no |
| y2_neg_2of3 | after days | 0.504 | 0.504 | -0.075 | 0.178 |  | YES |
| y3_recover_cash_6m | days after tax (inverse) | 0.684 | 0.682 | 0.035 | 0.178 |  | no |


## 4. Calendar — Q-peaked like missed-tax or monthly like salary?

Tax share Q-months 69.2% vs other 43.4% (gap 25.7%, ratio 1.59; tax_qa 69.2%/43.4% CONFIRM). Shape **q_peaked** (missed-tax Q-gap 25.9% LIKE missed-tax; salary Q-gap 0.6% not salary-monthly). Leftover after is_q 0.624; leftover after days on Q-months 0.547 on other 0.501.

| month | name | q | n | P(tax) |
| --- | --- | --- | --- | --- |
| 1 | Jan | 1 | 1,768 | 67.9% |
| 2 | Feb | 0 | 1,881 | 39.5% |
| 3 | Mar | 0 | 1,925 | 43.1% |
| 4 | Apr | 1 | 1,945 | 68.6% |
| 5 | May | 0 | 1,966 | 43.7% |
| 6 | Jun | 0 | 1,976 | 42.5% |
| 7 | Jul | 1 | 2,010 | 67.9% |
| 8 | Aug | 0 | 2,047 | 38.6% |
| 9 | Sep | 0 | 1,299 | 45.0% |
| 10 | Oct | 1 | 1,381 | 72.4% |
| 11 | Nov | 0 | 1,428 | 46.1% |
| 12 | Dec | 0 | 1,531 | 48.9% |


## 5. Dark vs ERP leftover

Last-month ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). P(tax) invoiced 52.5% dark 50.5%. Y3 leftover invoiced 0.472 dark 0.562.

| slice | n_cm | P(tax) | Y3 raw | leftover days |
| --- | --- | --- | --- | --- |
| invoiced_744 | 13,554 | 52.5% | 0.604 | 0.472 |
| dark_470 | 7,603 | 50.5% | 0.639 | 0.562 |


## 6. Q6 — lag leftover after days_lag1

Y3 tax now 0.611 lag1 0.597 lag3 0.597. Days lag1 0.684 (KEEP 0.684 CONFIRM). tax_lag1 leftover after days_lag1 0.519 lag3 0.518. Q6 CLOSE.

| col | n | n_pos | CV |
| --- | --- | --- | --- |
| c_tax_month | 5,648 | 402 | 0.611 |
| c_tax_month_lag1 | 5,648 | 402 | 0.597 |
| c_tax_month_lag3 | 5,078 | 355 | 0.597 |
| c_n_days_with_tx | 5,648 | 402 | 0.711 |
| c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 |


## 7. ICC / BETWEEN

c_tax_month ICC 0.927 (feature-report 0.93 CONFIRM BETWEEN). Y3 company-mean 0.739 demean 0.506 demean leftover after days 0.518 BETWEEN leftover after days-mean 0.645.

| y | feature | n | n_pos | CV | sign | folds |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | now | 5,648 | 402 | 0.611 | -1 | 0.556 0.624 0.630 0.613 0.630 |
| y3_recover_cash_6m | co_mean | 5,648 | 402 | 0.739 | -1 | 0.678 0.752 0.799 0.691 0.774 |
| y3_recover_cash_6m | demean | 5,648 | 402 | 0.506 | -1 | 0.466 0.516 0.499 0.543 0.507 |


## Extra — holdout coverage

Holdout coverage only (no fit): 72 co / 1,073 CM, cov 100.0% P(tax) 51.3%.

## Extra — fold-wise leftover

Y3 leftover-after-days rank folds 0.471 0.519 0.553 0.522 0.537 spread 0.082.

| fold | OLS | rank |
| --- | --- | --- |
| 0 | 0.471 | 0.471 |
| 1 | 0.519 | 0.519 |
| 2 | 0.553 | 0.553 |
| 3 | 0.522 | 0.522 |
| 4 | 0.537 | 0.537 |


Plot: `tax_month_qa.png`.

## What failed / next

- no replica miss; leftover after days is the unused-leftover decision

## Extra — leftover after Q dummy + days / ss

Leftover after Q+days 0.537; after Q+ss 0.533; after Q+days+ss+salary 0.515. Y3 raw Q-months 0.639 other 0.598.

## Extra — always / never / mixed cadence

Cadence always 157 / never 116 / mixed 941. Y3 rate always 1.5% never 15.6% mixed 8.0%. Mixed leftover after days 0.509.

| cadence | n_co | n_cm | n_y3 | Y3 rate | raw | leftover days |
| --- | --- | --- | --- | --- | --- | --- |
| always | 157 | 2651 | 953 | 1.5% | LOW_POWER | — |
| never | 116 | 1529 | 173 | 15.6% | LOW_POWER | — |
| mixed | 941 | 16977 | 4522 | 8.0% | 0.580 | 0.509 |


## Extra — so-far leftover

Leftover after days so_far<6 0.577; 6-11 0.457; ≥12 0.529; short 0.525.

| slice | n | n_pos | raw | leftover days |
| --- | --- | --- | --- | --- |
| so_far<6 | 1,436 | 89 | 0.621 | 0.577 |
| so_far 6-11 | 2,287 | 163 | 0.601 | 0.457 |
| so_far≥12 | 1,925 | 150 | 0.603 | 0.529 |
| short_<12 | 3,723 | 252 | 0.607 | 0.525 |


## Extra — tax vs SS XOR / leftover after ss+salary

Jaccard(tax, ss) 0.486 both 6,744. Disagree leftover after days 0.696. Leftover after ss+salary 0.515 (after ss alone 0.528).

## Extra — drop chronic

Drop chronic ('GROUP_0158', 'GROUP_0172'): leftover after days 0.518 raw 0.606.

## Extra — company bootstrap leftover after days

Company bootstrap n=280 leftover-after-days rank p05/p50/p95 0.458 / 0.520 / 0.554; share <0.55 93.2%; wall 17s.

## Extra — BETWEEN tax leftover after SS-mean

BETWEEN tax leftover after SS-mean 0.658; after days-mean+SS-mean 0.600; raw tax leftover after SS-mean+days 0.491.

## Extra — tax≠SS leftover (Q split)

Disagree leftover after days 0.696; Q — other 0.708; tax-only 0.646 ss-only —.

| slice | n | n_pos | raw | leftover days |
| --- | --- | --- | --- | --- |
| tax≠ss | 1,926 | 142 | 0.621 | 0.696 |
| tax≠ss & Q | 499 | 45 | LOW_POWER | — |
| tax≠ss & other | 1,427 | 97 | 0.641 | 0.708 |
| tax=1 ss=0 | 1,044 | 104 | 0.500 | 0.646 |
| tax=0 ss=1 | 882 | 38 | LOW_POWER | — |


## Extra — permutation null leftover after days

Within-fold permute tax leftover-after-days p05/p50/p95 0.589 / 0.611 / 0.631; observed 0.520; p(perm ≥ obs) 1.000; wall 7s.

## Extra — dark × Q leftover

Dark × Q leftover after days: invoiced Q 0.564; invoiced other 0.532; dark Q —; dark other 0.572

| slice | n_pos | raw | leftover days |
| --- | --- | --- | --- |
| invoiced Q | 89 | 0.655 | 0.564 |
| invoiced other | 175 | 0.580 | 0.532 |
| dark Q | 46 | LOW_POWER | — |
| dark other | 92 | 0.649 | 0.572 |


## Extra — Q dummy leftover after days

is_q_month Y3 0.495 leftover after days 0.621. tax leftover after Q+days 0.537.

## Extra — leave-one-group leftover after days

Leave-one-group leftover after days (top 30): min/med/max 0.514 / 0.520 / 0.528.

## Extra — bootstrap leftover after Q+days

Bootstrap leftover after Q+days p05/p50/p95 0.505 / 0.541 / 0.568; share<0.55 74.4%; wall 10s.

## Extra — SIZE tercile leftover after days

Leftover after days T1 0.504; T2+T3 0.515.

| slice | n_pos | raw | leftover days |
| --- | --- | --- | --- |
| T1 | 222 | 0.541 | 0.504 |
| T2+T3 | 180 | 0.587 | 0.515 |


## Extra — Y3 rate cells days tercile × tax

Y3 rate cells by days tercile × tax (sign −).

| days tercile | tax | n | n_pos | Y3 rate |
| --- | --- | --- | --- | --- |
| low days | 0 | 1,154 | 184 | 15.9% |
| low days | 1 | 729 | 77 | 10.6% |
| mid | 0 | 640 | 34 | 5.3% |
| mid | 1 | 1,242 | 61 | 4.9% |
| high days | 0 | 309 | 19 | 6.1% |
| high days | 1 | 1,574 | 27 | 1.7% |


## Extra — leftover after other C stems

Leftover after other C stems: c_n_tx 0.526; c_recency_days 0.546

| control | ρ | leftover |
| --- | --- | --- |
| c_n_tx | 0.429 | 0.526 |
| c_recency_days | -0.325 | 0.546 |


## Extra — leftover after days on Jan/Apr/Jul/Oct

Q-month leftover after days Jan — Apr — Jul — Oct —. Leftover after days+recency 0.518.

| month | n_pos | raw | leftover days |
| --- | --- | --- | --- |
| Jan | 45 | LOW_POWER | — |
| Apr | 20 | LOW_POWER | — |
| Jul | 37 | LOW_POWER | — |
| Oct | 33 | LOW_POWER | — |



Elapsed 42s.

Did **not**: overwrite `tax_qa.*` / `ss_qa.*` / `salary_qa.*` / `salary_month_qa.*` / `in3_qa.*` / `issued_qa.*`, edit `ops.py` / `gbm_core.py`, put tax on the 15-col card, grow TURNOVER, invent `y_tax`, run the assembler, write 0–100, fit holdout, touch `product/`.
