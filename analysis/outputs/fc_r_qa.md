# Unused leftover of KEEP-flow `f_fc_r`

Generated `2026-09-19T05:12:16+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_fc`. Do not merge Family M. Do not grow TURNOVER. Do not change the 15-col Y3 card.

`f_fc_r` = trailing-3m fee+interest / inflow (Family F). Already KEEP on the 44. `f_fc_r_lag3` is on Y7 TURNOVER. The unused question is leftover after `f_ds_r` / days as Y3 X, leftover after issued_lag1 as Y7, twin vs `f_ds_r` / `a_fin_cost` / in-memory `m_fin`, and whether contemporaneous `f_fc_r` should leave the 44 while lag3 stays.

## Headline

Contemporaneous `f_fc_r` on the 44: **DROP** (unused leftover, not a twin (ρ vs f_ds_r=0.205, R²=0.005): Y3 leftover after f_ds_r rank 0.464 / after days 0.449 both die; single 0.559 loses to size 0.617 and days 0.711). `f_fc_r_lag3` on TURNOVER: **KEEP — do not rip; TURNFC0/1 failed fold 4**. Y3 leftover after `f_ds_r` rank 0.464 (dies); after days 0.449. Y7 leftover after issued_lag1 0.563; after TURN_NOFC 0.557. Not a twin of `f_ds_r` (ρ=0.205); vs Y9 ρ=0.132 Jaccard(hi,Y9+)=0.113. Q6 lag3 empty until so-far≥6: CONFIRM. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Contemporaneous fc is a financing **flow**, not a new health Y. DROP as 44-col X. |
| 2 | Who is improving? | Not a path column by itself. |
| 3 | Who is turning? | Y9 forbids F — fc is the fee_r raw material. fee_r raw material (Y9 forbids F as X); not a binary twin (ρ 0.13). |
| 4 | Dip vs fall? | Y7 leftover after issued_lag1 / TURN_NOFC is the unused test. Do not grow 0.720. |
| 5 | Why did it change? | Not a twin of `f_ds_r` (ρ=0.205). Leftover after ds_r 0.464 dies — unused leftover of the KEEP flow. |
| 6 | Months earlier? | `f_fc_r_lag3` empty until so-far≥6. F lag3 **CLOSE** (already). |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| contemporaneous `f_fc_r` on the 44 | **DROP** | unused leftover, not a twin (ρ vs f_ds_r=0.205, R²=0.005): Y3 leftover after f_ds_r rank 0.464 / after days 0.449 both die; single 0.559 loses to size 0.617 and days 0.711 |
| `f_fc_r` as Y3 X | **DROP** | vs size 0.617 / days 0.711 / leftover-ds 0.464 |
| `f_fc_r_lag3` on TURNOVER | **KEEP** | KEEP — do not rip; TURNFC0/1 failed fold 4 |
| F lag3 as Q6 | **CLOSE** | empty until so-far≥6 (CONFIRM=True) |
| `f_fc_r` as Y9 X | **CLOSE** | Y9 META forbids F; ρ vs label 0.132 |
| `f_outstanding_gt_granted` | **PARK** | last-month-only=True; not a twin of fc (ρ=0.042) |
| Family M merge | **CLOSE** | in-memory only; do not merge |
| invent `y_fc` | **CLOSE** | Y9 already is sustained fee_r |
| Family F flow `f_fc_r` in the store | **KEEP** | observed fee/inflow trail; DROP is from the 44-col starter only |

KEEP-as-X for *adding*: oriented group-fold beats size ≥0.02 **and** leftover after the honest bar **and** not SIZE **and** not a twin. For a KEEP column, leftover dying after `f_ds_r` **and** |ρ|≥0.80 is a twin / rewrite. Leftover dying with low R² / low ρ is **unused leftover** — still DROP contemporaneous from the 44; do not rip lag3 off TURNOVER.

## 1. Coverage; 470 vs ERP; ever-n

Train 1,214 co / 21,157 CM. f_fc_r cov 88.5% vs f_ds_r 88.5% (SAME rolling-3 hole). Dark 470 vs ERP 744: fc nn CM 6,663 / 12,066. Access ≠ ERP — F is bank/debt, dark companies have f_fc_r.

| col | n_nn | cov CM | ever co | ever % | ever >0 | share>0 | p50 | share=0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f_fc_r | 18,729 | 88.5% | 1,214 | 100.0% | 1,026 | 71.2% | 0.0003 | 28.8% |
| f_ds_r | 18,729 | 88.5% | 1,214 | 100.0% | 491 | 28.8% | 0.0000 | 71.2% |


Dark vs ERP (F is bank/debt — access ≠ ERP):

| book | col | n_co | n_cm | n_nn | cov CM | ever co | share=0 | zero_fill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dark | f_fc_r | 470 | 7603 | 6663 | 87.6% | 470 | 32.9% | no — 0 is no-fee 3m, not a dark fill |
| dark | f_ds_r | 470 | 7603 | 6663 | 87.6% | 470 | 68.1% | no — 0 is no-fee 3m, not a dark fill |
| ERP | f_fc_r | 744 | 13554 | 12066 | 89.0% | 744 | 26.5% | no — 0 is no-fee 3m, not a dark fill |
| ERP | f_ds_r | 744 | 13554 | 12066 | 89.0% | 744 | 73.0% | no — 0 is no-fee 3m, not a dark fill |


## 2. Spearman twins

f_fc_r vs f_ds_r ρ=0.205 vs lag1 0.887 vs lag3 0.682 vs a_fin_cost 0.663 vs a_out6 0.141 vs m_fin 0.739 vs days 0.232 vs size 0.035 vs ogtg 0.042. Twins: ['f_fc_r_lag1']. SIZE=False. ogtg_twin=False.

| a | b | ρ | twin_|ρ|≥0.80 | SIZE_|ρ|≥0.50 |
| --- | --- | --- | --- | --- |
| f_fc_r | f_ds_r | 0.205 | no | — |
| f_fc_r | f_fc_r_lag1 | 0.887 | YES | — |
| f_fc_r | f_fc_r_lag3 | 0.682 | no | — |
| f_fc_r | a_fin_cost | 0.663 | no | — |
| f_fc_r | a_out6 | 0.141 | no | — |
| f_fc_r | m_fin_share | 0.739 | no | — |
| f_fc_r | m_fee_share | 0.644 | no | — |
| f_fc_r | log1p(a_in3) | 0.035 | no | no |
| f_fc_r | c_n_days_with_tx | 0.232 | no | — |
| f_fc_r | f_outstanding_gt_granted | 0.042 | no | — |


## 3. Single-feature train group-fold AUROC

Y3 f_fc_r CV 0.559 vs size 0.617 (CONFIRM 0.617) vs days 0.711 (CONFIRM 0.711) vs f_ds_r 0.620. beat_size=False. Y7 f_fc_r 0.555 lag3 0.571 vs issued_lag1 0.630 (CONFIRM ~0.630).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | f_fc_r | 5,528 | 391 | 0.559 | 0.541 | 0.084 | -1 | 0.550 0.541 0.703 0.500 0.500 |
| y3_recover_cash_6m | f_fc_r_lag1 | 5,078 | 355 | 0.464 | 0.535 | 0.114 | -1 | 0.568 0.499 0.270 0.481 0.503 |
| y3_recover_cash_6m | f_fc_r_lag3 | 4,212 | 313 | 0.452 | 0.508 | 0.094 | -1 | 0.553 0.463 0.304 0.437 0.503 |
| y3_recover_cash_6m | f_ds_r | 5,528 | 391 | 0.620 | 0.625 | 0.045 | -1 | 0.547 0.633 0.671 0.618 0.627 |
| y3_recover_cash_6m | e_ar_issued_lag1 | 3,618 | 264 | 0.671 | 0.669 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y7_top1_lost | f_fc_r | 7,000 | 1,987 | 0.555 | 0.550 | 0.076 | 1 | 0.559 0.674 0.561 0.499 0.482 |
| y7_top1_lost | f_fc_r_lag1 | 6,518 | 1,847 | 0.559 | 0.553 | 0.087 | 1 | 0.565 0.700 0.559 0.499 0.474 |
| y7_top1_lost | f_fc_r_lag3 | 5,481 | 1,496 | 0.571 | 0.567 | 0.087 | 1 | 0.571 0.709 0.573 0.523 0.478 |
| y7_top1_lost | f_ds_r | 7,000 | 1,987 | 0.521 | 0.523 | 0.027 | -1 | 0.549 0.502 0.514 0.491 0.551 |
| y7_top1_lost | e_ar_issued_lag1 | 7,253 | 2,072 | 0.630 | 0.623 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y7_top1_lost | log1p_a_in3 | 7,000 | 1,987 | 0.469 | 0.536 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.511 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 |
| y2_neg_2of3 | f_fc_r | 14,968 | 1,044 | 0.534 | 0.521 | 0.053 | 1 | 0.545 0.618 0.481 0.497 0.528 |
| y2_neg_2of3 | f_fc_r_lag1 | 13,776 | 938 | 0.535 | 0.519 | 0.062 | 1 | 0.546 0.631 0.475 0.488 0.534 |
| y2_neg_2of3 | f_fc_r_lag3 | 11,477 | 748 | 0.529 | 0.517 | 0.068 | 1 | 0.547 0.621 0.449 0.477 0.554 |
| y2_neg_2of3 | f_ds_r | 14,968 | 1,044 | 0.538 | 0.539 | 0.066 | 1 | 0.539 0.621 0.438 0.532 0.557 |
| y2_neg_2of3 | e_ar_issued_lag1 | 10,539 | 646 | 0.542 | 0.532 | 0.062 | 1 | 0.577 0.601 0.533 0.558 0.441 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.540 | 0.046 | 1 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.577 | 0.046 | 1 | 0.623 0.539 0.612 0.565 0.517 |


Night replicas: days 0.711 (MATCH 0.711), size 0.617 (MATCH 0.617), issued_lag1 0.630 (MATCH ~0.630).

## 4. Honest leftover after `f_ds_r` (Y3) and after days (Y3)

Y3 leftover after f_ds_r OLS 0.545 rank 0.464 R²=0.005 (dies <0.55 — unused leftover (low R², not a twin rewrite)). After days OLS 0.648 rank 0.449 (dies). After size 0.554. After ds+days 0.440. resid is not an fc clone.

| y | control | OLS | rank | ρ(resid,ctrl) | ρ(resid,fc) | R² | n | honest_dies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | after f_ds_r | 0.545 | 0.464 | -0.442 | 0.666 | 0.005 | 5,528 | YES |
| y3_recover_cash_6m | after days | 0.648 | 0.449 | 0.740 | 0.594 | 0.043 | 5,528 | YES |
| y3_recover_cash_6m | after size | 0.562 | 0.554 | 0.684 | 0.429 | 0.241 | 5,528 | no |
| y3_recover_cash_6m | after ds+days | 0.634 | 0.440 | 0.147 | 0.560 | 0.048 | 5,528 | YES |
| y3_recover_cash_6m | after a_fin_cost | 0.558 | 0.508 | 0.645 | 0.992 | 0.000 | 5,528 | YES |
| y2_neg_2of3 | after f_ds_r | 0.507 | 0.523 | -0.442 | 0.666 | 0.005 | 14,968 | YES |
| y2_neg_2of3 | after days | 0.572 | 0.457 | 0.740 | 0.594 | 0.043 | 14,968 | YES |
| y3_recover_cash_6m | after m_fin | 0.568 | 0.462 | 0.499 | 0.864 | 0.220 | 5,422 | YES |


Leftover <0.55 dies. Rank leftover is the honest bar (OLS can fake a days leak).

## 5. Y7 leftover after issued_lag1 and TURNOVER-without-fc

Y7 leftover after issued_lag1 OLS 0.551 rank 0.563 (lives (bare)). After TURNOVER-without-fc OLS 0.644 rank 0.557 ρ(resid,fc)=0.633 (resid is not an fc clone). lag3 after issued 0.579 after TURN_NOFC 0.576. Did not refit TURNOVER. Quote 0.720 / TURN_NOFC 0.714 unchanged. Bare leftover ≥0.55 is CLOSE as a TURNOVER add-on (TURNFC0 failed fold 4).

| residual | OLS | rank | R² | ρ(resid,fc) | n | n_pos | honest_dies | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contemp after issued_lag1 | 0.551 | 0.563 | 0.000 | 0.984 | 7,000 | 1,987 | no | 0.554 0.665 0.561 0.503 0.474 |
| contemp after days | 0.441 | 0.560 | 0.043 | 0.594 | 7,000 | 1,987 | no | 0.429 0.427 0.474 0.435 0.439 |
| lag3 after issued_lag1 | 0.564 | 0.579 | 0.000 | 0.986 | 5,481 | 1,496 | no | 0.562 0.700 0.575 0.512 0.472 |
| contemp after TURN_NOFC | 0.644 | 0.557 | 0.002 | 0.633 | 6,737 | 1,817 | no | 0.627 0.717 0.640 0.600 0.637 |
| lag3 after TURN_NOFC | 0.635 | 0.576 | 0.000 | 0.768 | 5,297 | 1,363 | no | 0.636 0.750 0.633 0.605 0.553 |


Single leftover only. Did not refit TURNOVER. TURN_NOFC quote 0.714 / TURNOVER 0.720 stand. TURNFC0 / TURNFC1 already failed fold 4 — do not grow.

## 6. Q6 — lag1 / lag3 empty-on-short

f_fc_r_lag3 nn on Y7 so-far<6: 0/1983 (CONFIRM empty until so-far≥6). Y7 so-far≥6 lag3 nn 5481. f_fc_r nn on Y7 so-far<3: 0 (rolling-3 hole). F lag3 Q6 stays CLOSE.

| y | bucket | n_lab | f_fc_r | f_fc_r_lag1 | f_fc_r_lag3 | f_ds_r | f_ds_r_lag3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | <6 | 1436 | 1,316 (91.6%) | 866 (60.3%) | 0 (0.0%) | 1,316 (91.6%) | 0 (0.0%) |
| y3_recover_cash_6m | 6-11 | 2287 | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) | 2,287 (100.0%) |
| y3_recover_cash_6m | 12-17 | 1712 | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) | 1,712 (100.0%) |
| y3_recover_cash_6m | 18-23 | 213 | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) |
| y3_recover_cash_6m | short_<12 | 3723 | 3,603 (96.8%) | 3,153 (84.7%) | 2,287 (61.4%) | 3,603 (96.8%) | 2,287 (61.4%) |
| y3_recover_cash_6m | long_>=18 | 213 | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) | 213 (100.0%) |
| y3_recover_cash_6m | all | 5648 | 5,528 (97.9%) | 5,078 (89.9%) | 4,212 (74.6%) | 5,528 (97.9%) | 4,212 (74.6%) |
| y7_top1_lost | <6 | 1983 | 1,519 (76.6%) | 1,037 (52.3%) | 0 (0.0%) |  |  |
| y7_top1_lost | 6-11 | 2438 | 2,438 (100.0%) | 2,438 (100.0%) | 2,438 (100.0%) |  |  |
| y7_top1_lost | 12-17 | 2127 | 2,127 (100.0%) | 2,127 (100.0%) | 2,127 (100.0%) |  |  |
| y7_top1_lost | 18-23 | 916 | 916 (100.0%) | 916 (100.0%) | 916 (100.0%) |  |  |
| y7_top1_lost | short_<12 | 4421 | 3,957 (89.5%) | 3,475 (78.6%) | 2,438 (55.1%) |  |  |
| y7_top1_lost | long_>=18 | 916 | 916 (100.0%) | 916 (100.0%) | 916 (100.0%) |  |  |
| y7_top1_lost | all | 7464 | 7,000 (93.8%) | 6,518 (87.3%) | 5,481 (73.4%) |  |  |


Short-book / so-far singles:

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Y7 short_<12 f_fc_r | f_fc_r | 3,957 | 1,175 | 0.534 | 0.519 | 0.089 | 1 | 0.556 0.668 0.536 0.464 0.443 |
| Y7 so-far<6 f_fc_r (lag3 empty) | f_fc_r | 1,519 | 491 | 0.431 | 0.512 | 0.061 | 1 | 0.418 0.338 0.503 0.459 0.436 |
| Y7 so-far≥6 f_fc_r | f_fc_r | 5,481 | 1,496 | 0.565 | 0.564 | 0.071 | 1 | 0.553 0.679 0.581 0.518 0.496 |
| Y7 so-far≥6 f_fc_r_lag3 | f_fc_r_lag3 | 5,481 | 1,496 | 0.571 | 0.567 | 0.087 | 1 | 0.571 0.709 0.573 0.523 0.478 |
| Y7 long f_fc_r_lag3 | f_fc_r_lag3 | 916 | 209 | 0.640 | 0.634 | 0.105 | 1 | 0.580 0.806 0.674 0.600 0.540 |
| Y3 short f_fc_r | f_fc_r | 3,603 | 241 | 0.441 | 0.535 | 0.098 | -1 | 0.476 0.537 0.280 0.427 0.485 |
| Y3 so-far<6 f_fc_r | f_fc_r | 1,316 | 78 | 0.344 | 0.521 | 0.077 | 1 | 0.347 0.386 0.351 0.218 0.420 |


## 7. SIZE terciles / ICC / demean

ICC 0.935 acf1 0.578 acf3 -0.134 (TRAIT). Y3 demean 0.600 vs company-mean 0.584. Y9 company-mean 0.584 demean 0.516 (not the m_fin trait-leak shape).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | f_fc_r | 5,528 | 391 | 0.559 | 0.541 | 0.084 | -1 | 0.550 0.541 0.703 0.500 0.500 |
| y3_recover_cash_6m | company-mean | 5,648 | 402 | 0.584 | 0.609 | 0.104 | 1 | 0.496 0.539 0.519 0.615 0.751 |
| y3_recover_cash_6m | company-demean | 5,528 | 391 | 0.600 | 0.617 | 0.085 | -1 | 0.542 0.540 0.550 0.636 0.734 |
| y7_top1_lost | f_fc_r | 7,000 | 1,987 | 0.555 | 0.550 | 0.076 | 1 | 0.559 0.674 0.561 0.499 0.482 |
| y7_top1_lost | company-mean | 7,464 | 2,149 | 0.529 | 0.530 | 0.060 | 1 | 0.472 0.617 0.546 0.534 0.475 |
| y7_top1_lost | company-demean | 7,000 | 1,987 | 0.469 | 0.503 | 0.028 | -1 | 0.465 0.467 0.507 0.429 0.476 |
| y9_fee_r_ownp80 | f_fc_r | 9,591 | 1,350 | 0.608 | 0.609 | 0.031 | 1 | 0.579 0.646 0.608 0.630 0.575 |
| y9_fee_r_ownp80 | company-mean | 9,591 | 1,350 | 0.584 | 0.588 | 0.041 | 1 | 0.528 0.599 0.618 0.622 0.555 |
| y9_fee_r_ownp80 | company-demean | 9,591 | 1,350 | 0.516 | 0.516 | 0.018 | 1 | 0.524 0.541 0.499 0.500 0.518 |


Within size terciles (hi/lo = above/below median `f_fc_r` in the tercile):

| T | y | n_lab | n_pos | rate | rate hi-fc | rate lo-fc |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | y3_recover_cash_6m | 1331 | 222 | 16.7% | 15.6% | 18.0% |
| T1 | y7_top1_lost | 1956 | 685 | 35.0% | 44.3% | 26.5% |
| T1 | y9_fee_r_ownp80 | 3413 | 375 | 11.0% | 16.2% | 5.5% |
| T2 | y3_recover_cash_6m | 2084 | 116 | 5.6% | 4.1% | 7.1% |
| T2 | y7_top1_lost | 2946 | 812 | 27.6% | 30.8% | 24.4% |
| T2 | y9_fee_r_ownp80 | 3285 | 536 | 16.3% | 20.2% | 12.0% |
| T3 | y3_recover_cash_6m | 2233 | 64 | 2.9% | 2.2% | 3.5% |
| T3 | y7_top1_lost | 2562 | 652 | 25.4% | 27.0% | 24.0% |
| T3 | y9_fee_r_ownp80 | 2893 | 439 | 15.2% | 16.5% | 13.8% |


## 8. vs Y9 accepted labels

f_fc_r vs Y9 own-p80 ρ=0.132 Jaccard(hi-p80,Y9+)=0.113 Jaccard(fc>0,Y9+)=0.168. Y9 META: f_fc_r is the fee_r raw material (forbidden F as Y9 X). Not a |ρ|≥0.80 / Jaccard≥0.50 rewrite of the binary Y9 label (raw material still forbidden as Y9 X).

| pair | n | ρ | is_Y_|ρ|≥0.80 |
| --- | --- | --- | --- |
| f_fc_r vs y9_fee_r_ownp80 | 9591 | 0.132 | no |
| f_fc_r vs y9_fee_spike | 7879 | 0.028 | no |


| set A | set B | Jaccard | n_A | n_B | n_both |
| --- | --- | --- | --- | --- | --- |
| f_fc_r > company p80 | Y9=1 | 0.113 | 1683 | 1350 | 307 |
| f_fc_r > 0 | Y9=1 | 0.168 | 7043 | 1350 | 1205 |


Y9 forbids family F as X. This is an identity check, not a Y9 model.

## 9. Fold 4 / short books

Fold 4 (sign 0–3): f_fc_r 0.482 lag3 0.478 issued_lag1 0.647. TURNOVER fold-4 quote 0.680. Y7 so-far<6 (lag3 empty) contemporaneous 0.431 vs issued 0.618. Contemporaneous does not rescue short books where lag3 is empty. Fold 4 is still issued / issued-CV, not contemporaneous fc.

| feature | CV | fold4 | folds |
| --- | --- | --- | --- |
| f_fc_r | 0.555 | 0.482 | 0.559 0.674 0.561 0.499 0.482 |
| f_fc_r_lag3 | 0.571 | 0.478 | 0.571 0.709 0.573 0.523 0.478 |
| e_ar_issued_lag1 | 0.630 | 0.647 | 0.643 0.662 0.590 0.605 0.647 |
| e_ar_issued_lag_cv | 0.627 | 0.605 | 0.609 0.666 0.600 0.654 0.605 |
| f_ds_r | 0.521 | 0.551 | 0.549 0.502 0.514 0.491 0.551 |


Fold 4 only (sign from folds 0–3):

| feature | n_va | n_pos | fold4 | sign |
| --- | --- | --- | --- | --- |
| f_fc_r | 1665 | 625 | 0.482 | 1 |
| f_fc_r_lag3 | 1318 | 484 | 0.478 | 1 |
| e_ar_issued_lag1 | 1734 | 657 | 0.647 | -1 |
| e_ar_issued_lag_cv | 1734 | 657 | 0.605 | 1 |


| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Y7 so-far<6 | f_fc_r | 1,519 | 491 | 0.431 | 0.512 | 0.061 | 1 | 0.418 0.338 0.503 0.459 0.436 |


Fold-4 groups (y7_core.md):

| group | n_lab | n_pos | rate | f_fc_r p50 | lag3 nn |
| --- | --- | --- | --- | --- | --- |
| GROUP_0222 | 336 | 241 | 71.7% | 0.0005 | 243 |
| GROUP_0108 | 204 | 130 | 63.7% | 0.0027 | 189 |


## 10. Dark 470

Dark 470 vs ERP 744: f_fc_r nn 6,663 vs 12,066. e_ar_issued on dark 1 (BOOK vs E stub). Access ≠ ERP. Y3 f_fc_r dark 0.586 ERP 0.483.

| book | n_co | n_cm | f_fc_r nn | f_fc_r cov | f_fc_r ever | e_ar_issued nn | Y3 labeled | Y7 labeled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dark | 470 | 7603 | 6663 | 87.6% | 470 | 1 | 2030 | 0 |
| ERP | 744 | 13554 | 12066 | 89.0% | 744 | 13554 | 3618 | 7464 |


## 11. `f_outstanding_gt_granted` (PARK snapshot — do not reopen)

ogtg last-month-only=True n_nn=1,214 ρ vs f_fc_r=0.042 (not a twin — PARK snapshot stands).

| col | n_nn | cov | share last-month | last-month only | ρ vs f_fc_r | twin |
| --- | --- | --- | --- | --- | --- | --- |
| f_outstanding_gt_granted | 1214 | 5.7% | 100.0% | YES | 0.042 | no |


## 12. Brief map + PARK / CLOSE / KEEP / DROP-from-44

| object | on 44 today | this ticket | TURNOVER | Q6 |
| --- | --- | --- | --- | --- |
| contemporaneous `f_fc_r` | KEEP flow | **DROP** from the 44 | not on TURNOVER (TURNFC0 failed fold 4) | rolling-3 empty until so-far≥3 |
| `f_fc_r_lag3` | derived, not a 44 stem | do not rip | **KEEP** | **CLOSE** empty until so-far≥6 |
| `f_ds_r` | KEEP + 15-col stem | untouched | TURNDSSWAP 0.712 not a swap | lag3 CLOSE |

## Extra — holdout coverage only

Holdout 72 coverage only: 1,073 CM, fc nn 929, lag3 nn 714. No AUROC. No percentile fit.

| split | n_co | n_cm | f_fc_r nn | f_fc_r cov | lag3 nn | dark ever fc |
| --- | --- | --- | --- | --- | --- | --- |
| holdout | 72 | 1073 | 929 | 86.6% | 714 | 32 |


## Extra — quintiles

Y3 / Y7 / Y9 rates by f_fc_r quintile (train, labeled).

| y | Q | n_lab | n_pos | rate | fc p50 |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | (-0.001, 0.000104] | 1881 | 180 | 9.6% | 0.0000 |
| y3_recover_cash_6m | (0.000104, 0.000863] | 1283 | 57 | 4.4% | 0.0003 |
| y3_recover_cash_6m | (0.000863, 0.00627] | 1322 | 62 | 4.7% | 0.0021 |
| y3_recover_cash_6m | (0.00627, 1.0] | 1042 | 92 | 8.8% | 0.0385 |
| y7_top1_lost | (-0.001, 0.000104] | 2513 | 573 | 22.8% | 0.0000 |
| y7_top1_lost | (0.000104, 0.000863] | 1652 | 521 | 31.5% | 0.0003 |
| y7_top1_lost | (0.000863, 0.00627] | 1571 | 496 | 31.6% | 0.0021 |
| y7_top1_lost | (0.00627, 1.0] | 1264 | 397 | 31.4% | 0.0385 |
| y9_fee_r_ownp80 | (-0.001, 0.000104] | 3617 | 291 | 8.0% | 0.0000 |
| y9_fee_r_ownp80 | (0.000104, 0.000863] | 2040 | 354 | 17.4% | 0.0003 |
| y9_fee_r_ownp80 | (0.000863, 0.00627] | 1916 | 324 | 16.9% | 0.0021 |
| y9_fee_r_ownp80 | (0.00627, 1.0] | 2018 | 381 | 18.9% | 0.0385 |


## Extra — zero vs intensity

Zero vs intensity: Y3 flag 0.573 Y7 flag 0.528.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | f_fc_r>0 | 5,528 | 391 | 0.573 | 0.564 | 0.079 | -1 | 0.542 0.606 0.696 0.518 0.503 |
| y3_recover_cash_6m | f_fc_r on gt0 | 4,533 | 274 | 0.524 | 0.539 | 0.057 | 1 | 0.486 0.622 0.482 0.527 0.505 |
| y7_top1_lost | f_fc_r>0 | 7,000 | 1,987 | 0.528 | 0.528 | 0.036 | 1 | 0.544 0.565 0.553 0.486 0.492 |
| y7_top1_lost | f_fc_r on gt0 | 5,332 | 1,594 | 0.550 | 0.537 | 0.085 | 1 | 0.530 0.699 0.514 0.523 0.485 |


## Extra — leftover after in-memory `m_fin`

Leftover after in-memory m_fin: Y3 OLS 0.568 rank 0.462 Y9 OLS 0.555 rank 0.535 R²=0.220. Do not merge M.

| y | OLS | rank | R² | honest_dies |
| --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 0.568 | 0.462 | 0.220 | YES |
| y9_fee_r_ownp80 | 0.555 | 0.535 | 0.220 | YES |


## Extra — monthly `a_fin_cost / a_in3`

f_fc_r vs monthly a_fin_cost/max(a_in3,1) ρ=0.768 (trailing-3 is not the monthly ratio). Y3 leftover after monthly ratio rank 0.456.

## Extra — leftover after `f_ds_r` by fold

Y3 leftover-after-ds fold spread 0.149.

| fold | n | n_pos | leftover after f_ds_r |
| --- | --- | --- | --- |
| 0 | 1288 | 53 | 0.528 |
| 1 | 685 | 92 | 0.576 |
| 2 | 1047 | 64 | 0.444 |
| 3 | 1342 | 78 | 0.593 |
| 4 | 1166 | 104 | 0.584 |


## Extra — does `f_ds_r` survive residualizing `f_fc_r`?

Y3 f_ds_r raw 0.620 leftover after f_fc_r rank 0.606 OLS 0.597 R²=0.005 (15-col stem still lives — fc is the unused twin-ish leftover).

## Extra — leftover on fc>0 support

Y3 on fc>0: raw 0.524 leftover-ds rank 0.560 leftover-days 0.576 n=4,533 (intensity leftover lives).

## Extra — Y3 fold 2 (0.703) groups

Fold 2 Y3 positives share in top group GROUP_0155 = 16.7%. CV without fold 2 0.502. not a single-group dummy.

| group | n_lab | n_pos | n_co | rate | AUROC −fc |
| --- | --- | --- | --- | --- | --- |
| GROUP_0155 | 78 | 11 | 10 | 14.1% | 0.903 |
| GROUP_0150 | 82 | 8 | 10 | 9.8% | 0.686 |
| GROUP_0142 | 184 | 7 | 20 | 3.8% | 0.288 |
| GROUP_0064 | 34 | 7 | 5 | 20.6% | 0.788 |
| GROUP_0095 | 61 | 5 | 10 | 8.2% | 0.884 |
| GROUP_0094 | 74 | 4 | 7 | 5.4% | 0.688 |
| GROUP_0073 | 7 | 4 | 3 | 57.1% | 0.500 |
| GROUP_0020 | 14 | 3 | 4 | 21.4% | 1.000 |


## Extra — dark ∧ `e_ar_issued` defined

Dark ∧ e_ar_issued defined: 1 CM / 1 co. BOOK vs Family E mismatch on a stub — invoice-only still NaN-not-0 for the 470.

| company | group | period | e_ar_issued | f_fc_r |
| --- | --- | --- | --- | --- |
| COMP_0962 | GROUP_0154 | 2026-02-01 | 0.0 | -0.0000 |


## Extra — holdout Q6 (coverage only)

Holdout Q6 coverage only: Y7 so-far<6 lag3 nn 0/151 (CONFIRM empty).

| split | Y7 n_lab so-far<6 | lag3 nn | empty | Y7 n_lab so-far≥6 | lag3 nn ≥6 |
| --- | --- | --- | --- | --- | --- |
| holdout | 151 | 0 | YES | 240 | 240 |


## Extra — Y9 is a forward path, not now-fee_r

Y9 is a forward path, not now-fee_r: contemp ρ=0.132 lag3 ρ=-0.012 m_fin ρ=0.150. On company-p80-hi months ρ=0.081. Not the binary Y.

| x | n | ρ vs Y9 |
| --- | --- | --- |
| f_fc_r | 9591 | 0.132 |
| f_fc_r_lag1 | 9591 | 0.051 |
| f_fc_r_lag3 | 9591 | -0.012 |
| m_fin_share | 9591 | 0.150 |
| a_fin_cost | 9591 | 0.155 |


## Extra — OLS leftover after days is a fake leak

Y3 leftover after days OLS 0.648 rank 0.449 ρ(resid,days)=0.740 ρ(resid,fc)=0.594. OLS leftover is a fake days leak — honest rank dies.

## Extra — reconstruct fc from Family A

f_fc_r vs rolling-3 a_fin_cost / a_in3 ρ=1.000 (SAME as Family A trail). Y3 leftover after recon rank 0.531.

## Extra — company-demean leftover after days

Y3 company-demean fc raw 0.600 leftover after days rank 0.589 OLS 0.658 ρ(resid,days)=0.521 after ds_r 0.598. Y7 demean leftover after issued 0.464. month shock leftover lives but loses to size.

## Extra — GROUP_0155 (fold-2 top)

GROUP_0155 is 11 train companies, Y3 rate 14.1% vs rest 7.0%. Fold-2 top group, not a 25% dummy.

| slice | n_co | n_lab | n_pos | rate | fc p50 | med log in3 |
| --- | --- | --- | --- | --- | --- | --- |
| GROUP_0155 | 11 | 78 | 11 | 14.1% | 0.0001 | 13.947 |
| rest | 1203 | 5570 | 391 | 7.0% | 0.0005 | 13.222 |


Plot: `fc_r_leftover.png`.

## What failed / next

- OLS leftover after days 0.648 is a fake leak; honest rank 0.449 dies
- Y3 single without fold 2 is 0.502 — the 0.559 mean is one-fold leftover

Elapsed 9s. Cuts: coverage/470, twins, singles, Y3 leftover ds/days, Y7 leftover issued/TURN_NOFC, Q6 so-far, ICC/demean, Y9 Jaccard, fold 4 / short, dark, ogtg, holdout, quintiles, zero, m_fin leftover, monthly ratio, fold leftover, ds-after-fc, fc>0 support, fold-2 groups, dark issued stub, holdout Q6, Y9 lead ρ, fake days-OLS, A-recon, demean-days, GROUP_0155.

Did **not**: merge parquet, invent `y_fc`, merge Family M / I / J, edit `gbm_y7_core.py` / `gbm_core.py` / `dso_qa.py` / `cust_hhi_qa.py` / `d_tx_qa.py`, rewrite duckdb, run `build_targets`, touch `product/`, write 0–100, change night Y3 0.762/0.752 or TURNOVER 0.720, grow the 15-col card, write the parent journal / LIVE / canvas.
