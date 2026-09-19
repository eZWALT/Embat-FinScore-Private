# Unused leftover of `a_n_tx` after `c_n_days_with_tx`

Generated `2026-09-19T05:20:59+02:00` by agent `c81e4b07`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage / mix only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_n_tx`. Do not edit `cashflow.py` / `ops.py` / the 15-col card. Y3 never B. Night Y3 **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 TURNOVER **0.72 / 0.712**. Do not quote a_out_vol 0.722 as the engine.

`a_n_tx` = COUNT(*) of transactions this month (Family A). `c_n_tx` = COUNT(*) (Family C). `c_n_days_with_tx` = unique calendar days. `c_gap_sd` is already DROPPED from the 44 as the weaker days twin (ρ −0.905 vs days, −0.866 vs `a_n_tx`). This lane does not overwrite `gap_sd_qa.*`.

## Headline

CLOSE leftover-after-days rank 0.538 (OLS 0.623, fake=False). Y3 single 0.703 vs days 0.711 vs size 0.617. Identity vs c_n_tx=True. SIZE=True. Inverse days-after-n_tx 0.564. 15-col card: CLOSE as a days rewrite / DROP from the card. Q6 CLOSE. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | a_n_tx is a count/size stem (ICC 0.969 BETWEEN), not a health Y. Do not invent `y_n_tx`. |
| 2 | Who is improving? | Y3 leftover after days 0.538 — CLOSE. The 45→65 engine is days 0.711, not n_tx. |
| 3 | Who is turning? | Y2 a_n_tx 0.598 vs legal days 0.571. A COUNT(*) twin is not a turning X. |
| 4 | Dip vs fall? | Not this count. Intensity leftover is the only dip-shaped extra. |
| 5 | Why did it change? | Night SHAP: a_n_tx #4 mean|SHAP| 0.125, perm ΔAUROC 0.030, univ −0.714, SIZE_STEM ρ_size 0.651. Days is SHAP #8 perm 0.001 but the 0.711 bar. This-run leftover after days rank 0.538 OLS 0.623 fake=False. Perm 0.030 is the GBM using n_tx as the count/size rewrite of days — not leftover skill after the 0.711 bar. Days stays the engine; n_tx is the stem the trees picked because it is the SIZE_STEM twin. |
| 6 | Months earlier? | lag1 0.682 / lag3 0.666 — CLOSE. Days lag1 0.684 stays the KEEP. |


## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| a_n_tx leftover after days (Y3 X) | **CLOSE** | rank 0.538 OLS 0.623 ρ(resid,days)=-0.434 fake=False |
| 15-col Y3 card stem | **CLOSE as a days rewrite / DROP from the card** | a_n_tx ≡ c_n_tx (COUNT(*) identity). Y3 leftover after days rank 0.538 OLS 0.623 dies (ρ vs days 0.938). Inverse: days leftover after n_tx 0.564 survives — keep the 0.711 bar. Parent should DROP a_n_tx from the 15-col card; days stays. |
| COUNT(*) identity vs c_n_tx | **YES** | 21,157/21,157 equal; leftover after c_n_tx 0.702 |
| Twin vs days / gap_sd | **YES** | ρ days 0.938 c_n_tx 1.000 gap -0.866 |
| SIZE vs log1p(a_in3) | **YES** | ρ=0.661 (gate ≥0.50); feature-report 0.706 vs |a_op_in| |
| Inverse: days leftover after n_tx | **thin rank / OLS-high** | rank 0.564 OLS 0.710 — 0.711 bar is not a rewrite of n_tx on OLS; honest rank is thin |
| Q6 lag1 / lag3 | **CLOSE** | contemporaneous lag1 0.682 looks KEEP, but leftover after days_lag1 0.534 dies and n_tx is a days twin — CLOSE |
| ICC / trait vs month shock | **BETWEEN trait** | ICC 0.969 (quote 0.97 CONFIRM) |
| SHAP/perm 0.030 vs days bar | **rewrite, not leftover** | leftover 0.538; perm n_tx 0.030 days 0.001 |
| Night quotes | **unchanged** | Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712 |


## 1. Coverage; acf1/acf3; size ρ

Train a_n_tx cov 100.0% (CONFIRM 100%) mean 114.199 p50 43.000 eq0 4.2%. acf1 0.223 (feature-report 0.22 CONFIRM) acf3 0.132 acf6 0.066 — LOW_PERSIST. ρ vs log1p(a_in3) 0.661 vs log1p(|a_op_in|) 0.706 (SIZE |ρ|≥0.50).

| slice | n_cm | n_co | nn | cov | eq0 | mean | p50 | p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train all | 21,157 | 1214 | 21,157 | 100.0% | 4.2% | 114.199 | 43.000 | 288.000 |


| acf1 | acf3 | acf6 | ρ log1p(a_in3) | ρ log1p(|a_op_in|) |
| --- | --- | --- | --- | --- |
| 0.223 | 0.132 | 0.066 | 0.661 | 0.706 |

## 2. Spearman twins + COUNT(*) identity

a_n_tx vs c_n_tx equal on 21,157/21,157 (max|Δ|=0.000000) — IDENTITY (both COUNT(*)). Spearman twins |ρ|≥0.80: ['c_n_days_with_tx', 'c_n_tx', 'c_gap_sd']. vs days 0.938 vs c_n_tx 1.000 vs gap_sd -0.866 vs log1p(a_in3) 0.661 (SIZE).

| vs | ρ | twin | |ρ|≥0.80 |
| --- | --- | --- | --- |
| c_n_days_with_tx | 0.938 | YES | YES |
| c_n_tx | 1.000 | YES | YES |
| c_gap_sd | -0.866 | YES | YES |
| log1p(a_in3) | 0.661 |  |  |
| log1p(|a_op_in|) | 0.706 |  |  |
| intensity | 0.941 |  | YES |
| log1p_n_tx | 1.000 |  | YES |


## 3. Single-feature train group-fold AUROC

Y3 a_n_tx 0.703 vs days 0.711 (night 0.711 CONFIRM) vs size 0.617 (0.617 CONFIRM, Δ 0.087) vs c_n_tx 0.703 (same-n identity skill) vs gap_sd 0.686. Y2 a_n_tx 0.598 vs days 0.571 (legal days 0.571 CONFIRM). Loses to the 0.711 days bar.

Sign from the train side of each fold. Days bar 0.711. Size `log1p(a_in3)` 0.617. Y2 legal single was days 0.571. Never Y7.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | c_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | c_gap_sd | 5,574 | 398 | 0.686 | 0.693 | 0.038 | 1 | 0.630 0.721 0.690 0.672 0.720 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | log1p_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | intensity | 5,536 | 372 | 0.680 | 0.689 | 0.040 | -1 | 0.642 0.723 0.657 0.655 0.722 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 0.601 | 0.044 | 1 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | c_n_tx | 17,356 | 1,271 | 0.598 | 0.601 | 0.044 | 1 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.577 | 0.046 | 1 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | c_gap_sd | 16,559 | 1,207 | 0.577 | 0.585 | 0.040 | -1 | 0.620 0.519 0.609 0.569 0.568 |
| y2_neg_2of3 | log1p(a_in3) | 14,968 | 1,044 | 0.552 | 0.540 | 0.046 | 1 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | log1p_n_tx | 17,356 | 1,271 | 0.598 | 0.601 | 0.044 | 1 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | intensity | 16,764 | 1,236 | 0.619 | 0.617 | 0.040 | 1 | 0.655 0.600 0.667 0.587 0.585 |


## 4. Honest leftover after days (OLS + rank-ortho)

Y3 leftover after days OLS 0.623 rank 0.538 ρ(resid,days)=-0.434 R²=0.263 (resid is not a days clone). Same-n leftover after c_n_tx rank 0.702 (R²=1.000). After gap_sd 0.615. Inverse: days leftover after a_n_tx rank 0.564 OLS 0.710 (rank thin ≥0.55 — OLS almost the raw 0.711 bar (same OLS-high pattern)). Honest leftover after days DIES.

| y | control | OLS | rank | ρ(resid,ctrl) | ρ(resid,n_tx) | R² | fake-days | honest_dies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | after days | 0.623 | 0.538 | -0.434 | -0.234 | 0.263 |  | YES |
| y3_recover_cash_6m | after c_n_tx | 0.546 | 0.702 | 0.352 | 0.352 | 1.000 |  | no |
| y3_recover_cash_6m | after gap_sd | 0.565 | 0.615 | -0.322 | 0.568 | 0.067 |  | no |
| y3_recover_cash_6m | after size | 0.573 | 0.656 | -0.134 | 0.465 | 0.096 |  | no |
| y3_recover_cash_6m | after days+c_n_tx | 0.614 | 0.588 | 0.391 | 0.189 | 1.000 |  | no |
| y2_neg_2of3 | after days | 0.471 | 0.583 | -0.434 | -0.234 | 0.263 |  | no |
| y3_recover_cash_6m | days after n_tx (inverse) | 0.710 | 0.564 | 0.797 | 0.919 | 0.263 |  | no |


## 5. Inverse leftover — days after `a_n_tx`

Days leftover after n_tx rank 0.564 OLS 0.710. The 0.711 bar survives — days is not a rewrite of n_tx.

## 6. SIZE terciles — leftover inside T1 and T2+T3

Y3 leftover after days inside T1 rank 0.445 (dies); T2+T3 0.468 (dies). If leftover dies in both clocks it is not a small-firm regularity.

| slice | n | n_pos | raw n_tx | leftover days | days | size | R² | dies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 1,331 | 222 | 0.617 | 0.445 | 0.594 | 0.488 | 0.263 | YES |
| T2 | 2,084 | 116 | 0.612 | 0.475 | 0.613 | 0.557 | 0.263 | YES |
| T3 | 2,233 | 64 | 0.495 | 0.519 | 0.624 | 0.612 | 0.263 | YES |
| T2+T3 | 4,317 | 180 | 0.634 | 0.468 | 0.653 | 0.414 | 0.263 | YES |


## 7. Q6 — lag1 / lag3 on short books

Y3 a_n_tx now 0.703 lag1 0.682 lag3 0.666; short_<12 lag1 0.679. Days lag1 0.684 (night KEEP 0.684 CONFIRM). Q6 KEEP.

| y | slice | col | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | all | a_n_tx | 5,648 | 402 | 0.703 | -1 |
| y3_recover_cash_6m | all | a_n_tx_lag1 | 5,648 | 402 | 0.682 | -1 |
| y3_recover_cash_6m | all | a_n_tx_lag3 | 5,078 | 355 | 0.666 | -1 |
| y3_recover_cash_6m | all | c_n_days_with_tx | 5,648 | 402 | 0.711 | -1 |
| y3_recover_cash_6m | all | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | -1 |
| y3_recover_cash_6m | all | c_n_days_with_tx_lag3 | 5,078 | 355 | 0.666 | -1 |
| y3_recover_cash_6m | short_<12 | a_n_tx | 3,723 | 252 | 0.689 | -1 |
| y3_recover_cash_6m | short_<12 | a_n_tx_lag1 | 3,723 | 252 | 0.679 | -1 |
| y3_recover_cash_6m | short_<12 | a_n_tx_lag3 | 3,153 | 205 | 0.683 | -1 |
| y3_recover_cash_6m | short_<12 | c_n_days_with_tx | 3,723 | 252 | 0.696 | -1 |
| y3_recover_cash_6m | short_<12 | c_n_days_with_tx_lag1 | 3,723 | 252 | 0.684 | -1 |
| y3_recover_cash_6m | short_<12 | c_n_days_with_tx_lag3 | 3,153 | 205 | 0.691 | -1 |
| y3_recover_cash_6m | long_>=18 | a_n_tx | 213 | 16 | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | a_n_tx_lag1 | 213 | 16 | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | a_n_tx_lag3 | 213 | 16 | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | c_n_days_with_tx | 213 | 16 | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | c_n_days_with_tx_lag1 | 213 | 16 | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | c_n_days_with_tx_lag3 | 213 | 16 | LOW_POWER | — |
| y2_neg_2of3 | all | a_n_tx | 17,356 | 1,271 | 0.598 | 1 |
| y2_neg_2of3 | all | a_n_tx_lag1 | 16,161 | 1,151 | 0.601 | 1 |
| y2_neg_2of3 | all | a_n_tx_lag3 | 13,776 | 938 | 0.609 | 1 |
| y2_neg_2of3 | all | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 1 |
| y2_neg_2of3 | all | c_n_days_with_tx_lag1 | 16,161 | 1,151 | 0.575 | 1 |
| y2_neg_2of3 | all | c_n_days_with_tx_lag3 | 13,776 | 938 | 0.583 | 1 |
| y2_neg_2of3 | short_<12 | a_n_tx | 11,186 | 885 | 0.570 | 1 |
| y2_neg_2of3 | short_<12 | a_n_tx_lag1 | 9,991 | 765 | 0.569 | 1 |
| y2_neg_2of3 | short_<12 | a_n_tx_lag3 | 7,606 | 552 | 0.571 | 1 |
| y2_neg_2of3 | short_<12 | c_n_days_with_tx | 11,186 | 885 | 0.544 | 1 |
| y2_neg_2of3 | short_<12 | c_n_days_with_tx_lag1 | 9,991 | 765 | 0.545 | 1 |
| y2_neg_2of3 | short_<12 | c_n_days_with_tx_lag3 | 7,606 | 552 | 0.546 | 1 |
| y2_neg_2of3 | long_>=18 | a_n_tx | 1,908 | 119 | 0.696 | 1 |
| y2_neg_2of3 | long_>=18 | a_n_tx_lag1 | 1,908 | 119 | 0.704 | 1 |
| y2_neg_2of3 | long_>=18 | a_n_tx_lag3 | 1,908 | 119 | 0.711 | 1 |
| y2_neg_2of3 | long_>=18 | c_n_days_with_tx | 1,908 | 119 | 0.680 | 1 |
| y2_neg_2of3 | long_>=18 | c_n_days_with_tx_lag1 | 1,908 | 119 | 0.682 | 1 |
| y2_neg_2of3 | long_>=18 | c_n_days_with_tx_lag3 | 1,908 | 119 | 0.697 | 1 |


## 8. ICC / company-demean

a_n_tx ICC 0.969 (feature-report 0.97 CONFIRM BETWEEN); days ICC 0.985. TRAIT (BETWEEN). Y3 company-mean 0.694 demean 0.487 — the skill is a company identity, not a month shock.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | now | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | co_mean | 5,648 | 402 | 0.694 | 0.700 | 0.030 | -1 | 0.659 0.731 0.689 0.675 0.717 |
| y3_recover_cash_6m | demean | 5,648 | 402 | 0.487 | 0.502 | 0.026 | -1 | 0.493 0.521 0.457 0.499 0.463 |
| y2_neg_2of3 | now | 17,356 | 1,271 | 0.598 | 0.601 | 0.044 | 1 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | co_mean | 17,356 | 1,271 | 0.631 | 0.636 | 0.028 | 1 | 0.646 0.586 0.654 0.620 0.649 |
| y2_neg_2of3 | demean | 17,356 | 1,271 | 0.539 | 0.541 | 0.048 | -1 | 0.492 0.525 0.531 0.527 0.621 |


## 9. SHAP / perm gap

Night SHAP: a_n_tx #4 mean|SHAP| 0.125, perm ΔAUROC 0.030, univ −0.714, SIZE_STEM ρ_size 0.651. Days is SHAP #8 perm 0.001 but the 0.711 bar. This-run leftover after days rank 0.538 OLS 0.623 fake=False. Perm 0.030 is the GBM using n_tx as the count/size rewrite of days — not leftover skill after the 0.711 bar. Days stays the engine; n_tx is the stem the trees picked because it is the SIZE_STEM twin.

## Extra — same-n identity, log1p, intensity, burst

Same-n leftover after c_n_tx rank 0.702 R²=1.000 (lives). After n_tx+days 0.699. log1p(n_tx) raw 0.703 leftover-days 0.538. intensity raw 0.680 leftover-days 0.549. burst leftover 0.542.

| feature | OLS | rank | R² | dies |
| --- | --- | --- | --- | --- |
| a_n_tx after c_n_tx | 0.546 | 0.702 | 1.000 | no |
| a_n_tx after n_tx+days | 0.572 | 0.699 | 1.000 | no |
| log1p(n_tx) after days | 0.534 | 0.538 | 0.847 | YES |
| n_tx/days intensity after days | 0.561 | 0.549 | 0.002 | YES |
| n_tx−days burst after days | 0.623 | 0.542 | 0.234 | YES |


## Extra — 12-name Y2 drop (GROUP_0158 / 0172)

Chronic 12 names (0158/0172): 12. Y2 a_n_tx 0.598 → drop-12 0.582; days 0.571 → 0.549. Y3 0.703 → 0.700; leftover after days 0.538. Drop does not flip Y2 (≥0.03).

## Extra — holdout coverage only

Holdout coverage only (no fit): 72 companies / 1,073 CM, a_n_tx cov 100.0%, ρ vs days 0.915, ever-ERP 40 / dark 32. Not used to fit.

## Extra — dark 470 vs invoiced 744

Last-month companies ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Y3 leftover after days on invoiced 0.527 on dark 0.546.

| slice | n | n_pos | raw | leftover days | days | dies |
| --- | --- | --- | --- | --- | --- | --- |
| invoiced_744 | 3,618 | 264 | 0.725 | 0.527 | 0.730 | YES |
| dark_470 | 2,030 | 138 | 0.674 | 0.546 | 0.704 | YES |


## Extra — fold-wise leftover

Y3 leftover-after-days fold spread 0.112; folds 0.683 0.571 0.631 0.632 0.596.

| fold | n_tx | days | leftover |
| --- | --- | --- | --- |
| 0 | 0.652 | 0.665 | 0.683 |
| 1 | 0.732 | 0.738 | 0.571 |
| 2 | 0.697 | 0.700 | 0.631 |
| 3 | 0.699 | 0.715 | 0.632 |
| 4 | 0.737 | 0.740 | 0.596 |


## Extra — `a_n_tx_lag1` / `_lag3` after days_lag1

a_n_tx_lag1 leftover after days_lag1 rank 0.534 (dies); lag3 after days_lag1 0.532 (dies). Days lag1 is already KEEP (0.684 vs 0.711) — n_tx lags are rewrites if leftover dies.

| spec | raw | OLS | rank | R² | dies |
| --- | --- | --- | --- | --- | --- |
| a_n_tx_lag1 after days_lag1 | 0.682 | 0.607 | 0.534 | 0.261 | YES |
| a_n_tx_lag3 after days_lag1 | 0.666 | 0.609 | 0.532 | 0.219 | YES |
| a_n_tx_lag3 after days_lag3 | 0.666 | 0.602 | 0.532 | 0.258 | YES |
| a_n_tx after days_lag1 | 0.703 | 0.578 | 0.595 | 0.234 | no |


## Extra — identity residual dust

Identity residual a_n_tx~c_n_tx: R²=1.000000 max|resid|=2.84e-14 std=7.13e-15 unique@1e-12=1. Rank leftover 0.702 is numerical dust — not leftover skill (R²=1 identity).

## Extra — OLS 0.623 vs rank 0.538 (days+size)

OLS leftover after days 0.623 vs rank 0.538 ρ(resid,days)=-0.434 (OLS-high / rank-dies — treat rank as honest; not a |ρ|≥0.80 fake-days clone). After days+size rank 0.535 OLS 0.626.

| control | OLS | rank | ρ(resid,days) | R² |
| --- | --- | --- | --- | --- |
| after days | 0.623 | 0.538 | -0.434 | 0.263 |
| after days+size | 0.626 | 0.535 | -0.421 | 0.263 |


## Extra — fold 0 leftover

Fold-0 leftover 0.683 on 32 groups (top GROUP_0158 n=339 / 1310 labeled). Drop fold 0 leftover rank 0.544 OLS 0.608. not a single-group dummy.

## Extra — multi-tx months (n_tx>days)

Multi-tx months (n_tx>days) 87.7% of train. Y3 leftover after days on multi 0.488 on one-tx-per-day 0.495.

| slice | n_cm | n_y3 | raw | leftover | R² | dies |
| --- | --- | --- | --- | --- | --- | --- |
| n_tx>days | 18,552 | 5,347 | 0.662 | 0.488 | 0.263 | YES |
| n_tx==days | 2,605 | 301 | 0.495 | 0.495 | 0.263 | YES |


## Extra — Y2 leftover after days

Y2 leftover after days rank 0.583 OLS 0.471 drop-12 0.594. Intensity raw 0.619 leftover 0.597. Y2 leftover lives — not the 15-col Y3 card.

## Extra — company-demean leftover after days

Y3 company-demean a_n_tx raw 0.487 leftover after days rank 0.556 OLS 0.638 ρ(resid,days)=-0.336. month shock leftover lives.

## Extra — days leftover after n_tx+size

Days leftover after n_tx rank 0.564 OLS 0.710; after size 0.667; after both 0.562 (0.711 bar still has leftover).

| control | OLS | rank | R² |
| --- | --- | --- | --- |
| after n_tx | 0.710 | 0.564 | 0.263 |
| after size | 0.666 | 0.667 | 0.336 |
| after n_tx+size | 0.669 | 0.562 | 0.459 |


## Extra — short books + holdout mix

Y3 leftover after days on short_<12 rank 0.532 OLS 0.623 n_pos=252. Holdout mix (no fit): mean n_tx 121.826 p50 51.000 vs train mean 114.199 p50 43.000.

## Extra — rank-ortho fold leftover

Rank-ortho leftover after days folds 0.511 0.591 0.538 0.511 0.537 (mean 0.538); OLS folds 0.683 0.571 0.631 0.632 0.596 (mean 0.623). Inverse rank folds 0.560 0.533 0.564 0.582 0.580.

| direction | OLS folds | rank folds |
| --- | --- | --- |
| n_tx after days | 0.683 0.571 0.631 0.632 0.596 | 0.511 0.591 0.538 0.511 0.537 |
| days after n_tx | 0.700 0.717 0.707 0.698 0.729 | 0.560 0.533 0.564 0.582 0.580 |


## Extra — drop GROUP_0158/0172 from Y3 leftover

Drop GROUP_0158/0172: Y3 a_n_tx 0.701 days 0.708 leftover after days 0.538 OLS 0.630.

## Extra — demean after days+size / company-mean leftover

Demean leftover after days+size rank 0.550 OLS 0.634. Company-mean leftover after days rank 0.597 (trait rewrite of days).

## Extra — Y2 intensity after days+size; stressed SIZE ρ

Y2 intensity leftover after days+size rank 0.598. SIZE ρ a_n_tx vs log1p(a_in3) all 0.661 stressed-Y3 0.560 (importances 0.651 off).

## Extra — n_tx lags after contemporaneous days

a_n_tx_lag1 leftover after contemporaneous days rank 0.486; lag3 0.516.

## Extra — SHAP ρ_size hunt + company-median leftover

SHAP ρ_size hunt: all-in3 0.661 Y3-in3 0.560 Y3-opin 0.651 (hit 0.651). Company-median n_tx vs days ρ=0.939 vs size 0.694; median leftover after median-days 0.573.

| slice | ρ |
| --- | --- |
| all log_in3 | 0.661 |
| Y3 log_in3 | 0.560 |
| Y3 log_op_in | 0.651 |
| Y3 a_in3 | 0.560 |


## Extra — Y2 intensity KEEP-as-X (not the 15-col card)

Y2 intensity raw 0.619 vs size 0.552 (Δ 0.067) leftover after days 0.597 ρ vs days 0.781 vs size 0.618. fails KEEP-as-X even on Y2.

## Extra — drop zeros / long books / Y3 intensity

Y3 leftover after days, drop zeros: 0.532. Long ≥18m —. Y3 intensity leftover after days+size 0.541.

## Extra — group ICC + leftover after gap_sd

Company-median a_n_tx ICC across group_id 0.765 (k=235). Leftover after dropped twin gap_sd rank 0.615 (lives — n_tx is stronger than the dropped twin).

## Extra — leftover by months-so-far

Y3 leftover after days by months-so-far: <6 0.595, 6-11 0.456, 12-17 0.535, >=18 —.

| so_far | raw | leftover | n_pos |
| --- | --- | --- | --- |
| <6 | 0.656 | 0.595 | 89 |
| 6-11 | 0.710 | 0.456 | 163 |
| 12-17 | 0.669 | 0.535 | 134 |
| >=18 | LOW_POWER | — | 16 |


## Extra — so-far power + <6 after days+size

so-far <6 leftover after days+size 0.580 (pocket 0.595 after days only). Long books LOW_POWER if n_pos<50 — not a late-trail KEEP. Short-book leftover is not a 15-col stem.

| slice | n_lab | n_pos | n_tx leftover | days leftover | n_tx after days+size | LOW_POWER |
| --- | --- | --- | --- | --- | --- | --- |
| <6 | 1,436 | 89 | 0.595 | 0.433 | 0.580 |  |
| 6-11 | 2,287 | 163 | 0.456 | 0.620 | 0.467 |  |
| 12-17 | 1,712 | 134 | 0.535 | 0.544 | 0.544 |  |
| >=18 | 213 | 16 | — | — | — | yes |
| long_trail | 213 | 16 | — | — | — | yes |


## Extra — so-far <6 vs the days bar

so-far<6 Y3 n_tx 0.656 days 0.622 size 0.635 ρ(n_tx,days)=0.909 leftover 0.595 Δ vs days 0.034. n_tx beats days on the short pocket.

Plot: `n_tx_qa.png`.

## What failed / next

- leftover after c_n_tx rank 0.702 is identity dust (max|resid|=2.8e-14)
- OLS leftover after days 0.623 is high vs honest rank 0.538 (ρ(resid,days)=-0.434 < 0.80 — not a fake-days clone, still rank-dies)

Elapsed 9s. Cuts: coverage/acf/size ρ, twins+identity, singles (days 0.711 / size 0.617 CONFIRM), leftover after days, inverse leftover, SIZE terciles, Q6 short books, ICC/demean, SHAP/perm, same-n / log1p / intensity, 12-name Y2, holdout coverage, dark 470, fold leftover, lag leftovers.

Did **not**: rewrite `cashflow.py` / `ops.py`, overwrite `gap_sd_qa.*`, edit the 15-col card, grow TURNOVER, invent `y_n_tx`, merge Family I/M/J, rewrite `brief_map.md`, touch `product/`, write 0–100, fit holdout, run a new GBM, write the parent journal / LIVE / canvas.
