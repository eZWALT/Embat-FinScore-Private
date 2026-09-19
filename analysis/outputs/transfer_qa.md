# Signed `a_transfer` / `a_invest` — wash, quiet twin, or leftover?

Generated `2026-09-19T04:21:41+02:00` by agent `1bc809f3`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_transfer` / `y_invest`. Do not put `a_transfer` / `a_invest` on tonight's 15-col card. Night Y3 quote stays **0.762 / 0.752**. Do not merge Family M or I. `c_salary_month` 0.671 stays on the card. `c_missed_salary` CLOSE — not rewritten.

`a_transfer` = signed net `sum(amount | category=transfer)`. `a_invest` = signed `investment_deployment` + `investment_return`. `m_xfer_share` is |transfer|/|all| computed **in-module** (catmix.py not edited).

## Headline

Any-transfer 39.0%; any-invest 10.5%. Wash **NO** (|net|/gross p50 1.000). Y3 `a_transfer` 0.566 vs size 0.617 (Δ -0.051) vs days 0.711 vs salary 0.671. ρ vs days 0.109. Honest leftover has_xfer|days 0.579. ICC 0.956. X **CLOSE**. PARK as Y. `a_invest` X **CLOSE**. Q6 **CLOSE**.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Not a new Y. PARK `y_transfer` / `y_invest`. Any-transfer 39.0% of train CM. |
| 2 | Who is improving? | Signed transfer is not a recovery path. Quiet months recover — already on the card. |
| 3 | Who is turning? | Y3 `a_transfer` 0.566 vs size 0.617 / days 0.711 / salary 0.671. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | **CLOSE** — no sentence beyond quiet-stressed recover (already on the 15-col card). |
| 6 | Months earlier? | lag1/lag3 **CLOSE** — Y3 contemporaneous 0.566 loses to size 0.617 / days 0.711 (lag1 0.557 / lag3 0.542). CLOSE as Q6 — no useful contemporaneous skill to lead. SHAP lag1 perm-light CONFIRMED. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `a_transfer` as Y3 X | **CLOSE** | quiet twin of days/n_tx/salary (leftover dies); Y3 0.566 loses to size 0.617 / days 0.711 / salary 0.671. Explains SHAP-heavy / perm-light. Still **not** on tonight's 15-col card. |
| `a_transfer` as a health Y | **PARK** | do not invent `y_transfer` |
| `a_invest` as Y3 X | **CLOSE** | Y3 0.505 vs size 0.617; has_inv leftover after days 0.607; not G rise-only. Not on the card. |
| `a_invest` as a health Y | **PARK** | do not invent `y_invest` |
| two-way wash | **NO** | Train transfer CM 8,242: both-ways 48.0%, one-way (|net|≈gross) 55.5%, |net|/gross ≤0.05 9.4%. |net|/gross p10/p50/p90 0.057 / 1.000 / 1.000. Spearman signed↔gross 0.278; |signed|↔gross 0.967; signed↔|signed| 0.300. NO — typical transfer month is one-way (p50 |net|/gross = 1); signed is not a wash dummy. |
| quiet twin days/n_tx/salary | **YES** | signed ρ days 0.109 n_tx 0.097 salary 0.143; honest has_xfer leftover after days 0.579 (signed OLS leftover 0.666 is −days leak, ρ=-0.881) |
| SIZE | **NO** | ρ vs log1p(a_in3) 0.044 |
| Q6 lag1/lag3 | **CLOSE** | Y3 contemporaneous 0.566 loses to size 0.617 / days 0.711 (lag1 0.557 / lag3 0.542). CLOSE as Q6 — no useful contemporaneous skill to lead. SHAP lag1 perm-light CONFIRMED. |
| KEEP-Q5 footnote | **CLOSE** | no sentence beyond quiet-stressed recover (already on the 15-col card). |
| Family M / I | **do not merge** | `m_xfer_share` is |transfer|/|all|, not signed. I CLOSE already. |

## 1. Prevalence (train rates; holdout coverage)

Train any-transfer CM 39.0% (n=8,242 / 21,157); ever-transfer companies 882/1214. signed `a_transfer` p50=0 p90=127,950 (zero share 62.4%). Any-invest 10.5% (n=2,231); ever-invest 402/1214; signed p50=0 p90=0. Holdout coverage only: xfer 31.9% / invest 14.3% on 1,073 CM / 72 cos. No AUROC on holdout.

| split | col | n_cm | n_co | any | zero | p50 signed | p90 signed | p90 |signed| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | a_transfer | 21,157 | 1,214 | 39.0% | 62.4% | 0 | 127,950 | 249,737 |
| train | a_invest | 21,157 | 1,214 | 10.5% | 89.6% | 0 | 0 | 3 |
| holdout | a_transfer | 1,073 | 72 | 31.9% | 68.8% | 0 | 86,547 | 196,967 |
| holdout | a_invest | 1,073 | 72 | 14.3% | 86.3% | 0 | 0 | 2,335 |


Ever-transfer companies 882/1214; ever-invest 402/1214.

## 2. Tokens vs store

Store `a_transfer` vs raw sum(amount|category=transfer) max|Δ|=1.94e-06; `a_invest` vs deploy+return max|Δ|=7.45e-09. Formula OK — not a cashflow.py bug. Tokens present: transfer / investment_deployment / investment_return.

| category | n_tx | n_in | n_out | p50 |amt| |
| --- | --- | --- | --- | --- |
| investment_deployment | 3,923 | 0 | 3,923 | 1,470 |
| investment_return | 3,458 | 3,458 | 0 | 634 |
| transfer | 152,102 | 90,723 | 61,379 | 6,722 |


Sample company-months (store signed vs raw in / out / gross):

| company_id | period | store | in | out | gross |
| --- | --- | --- | --- | --- | --- |
| COMP_0001 | 2026-04 | 18,500.00 | 18,500.00 | 0.00 | 18,500.00 |
| COMP_0003 | 2026-04 | 0.00 | 1,091,836.00 | 1,091,836.00 | 2,183,672.00 |
| COMP_0003 | 2026-05 | -11,247.09 | 4,550,285.90 | 4,561,532.99 | 9,111,818.89 |
| COMP_0003 | 2026-06 | -48,388.75 | 5,343,637.98 | 5,392,026.73 | 10,735,664.71 |
| COMP_0001 | 2026-01 | 0.00 | 0.00 | 0.00 | 0.00 |
| COMP_0001 | 2026-02 | 0.00 | 0.00 | 0.00 | 0.00 |


## 3. Two-way wash — gross vs signed net

Train transfer CM 8,242: both-ways 48.0%, one-way (|net|≈gross) 55.5%, |net|/gross ≤0.05 9.4%. |net|/gross p10/p50/p90 0.057 / 1.000 / 1.000. Spearman signed↔gross 0.278; |signed|↔gross 0.967; signed↔|signed| 0.300. NO — typical transfer month is one-way (p50 |net|/gross = 1); signed is not a wash dummy.

| item | value |
| --- | ---: |
| transfer CM | 8,242 |
| both-ways share | 48.0% |
| one-way share | 55.5% |
| washed (|net|/gross ≤0.05) | 9.4% |
| \|net\|/gross p10 / p50 / p90 | 0.057 / 1.000 / 1.000 |
| ρ signed ↔ gross | 0.278 |
| ρ \|signed\| ↔ gross | 0.967 |
| ρ signed ↔ \|signed\| | 0.300 |

Plot: `transfer_wash.png`.

## 4. Spearman vs days / n_tx / salary / size / mix

Twin |ρ|≥0.80; SIZE |ρ| vs log1p(a_in3) ≥0.50. `a_transfer` vs days 0.109, n_tx 0.097, salary 0.143, ss 0.145, size 0.044, a_op_in 0.020, m_xfer_share 0.270. `has_xfer` vs days 0.448. `a_invest` vs size -0.018. Twins: abs_xfer↔m_xfer_share 0.935; abs_xfer↔xfer_gross 0.967; abs_xfer↔has_xfer 0.937; xfer_gross↔m_xfer_share 0.966; xfer_gross↔abs_xfer 0.967; xfer_gross↔has_xfer 0.961; has_xfer↔m_xfer_share 0.957; has_xfer↔xfer_gross 0.961; has_xfer↔abs_xfer 0.937. Signed net is not SIZE.

| object | vs | ρ | flag |
| --- | --- | --- | --- |
| a_transfer | c_n_days_with_tx | 0.109 |  |
| a_transfer | a_n_tx | 0.097 |  |
| a_transfer | c_salary_month | 0.143 |  |
| a_transfer | c_ss_month | 0.145 |  |
| a_transfer | log1p(a_in3) | 0.044 |  |
| a_transfer | a_op_in | 0.020 |  |
| a_transfer | m_xfer_share | 0.270 |  |
| abs_xfer | c_n_days_with_tx | 0.450 |  |
| abs_xfer | a_n_tx | 0.467 |  |
| abs_xfer | c_salary_month | 0.348 |  |
| abs_xfer | c_ss_month | 0.360 |  |
| abs_xfer | log1p(a_in3) | 0.372 |  |
| abs_xfer | a_op_in | 0.373 |  |
| abs_xfer | m_xfer_share | 0.935 | TWIN |
| has_xfer | c_n_days_with_tx | 0.448 |  |
| has_xfer | a_n_tx | 0.459 |  |
| has_xfer | c_salary_month | 0.357 |  |
| has_xfer | c_ss_month | 0.372 |  |
| has_xfer | log1p(a_in3) | 0.346 |  |
| has_xfer | a_op_in | 0.351 |  |
| has_xfer | m_xfer_share | 0.957 | TWIN |
| a_invest | c_n_days_with_tx | 0.021 |  |
| a_invest | a_n_tx | 0.015 |  |
| a_invest | c_salary_month | -0.015 |  |
| a_invest | c_ss_month | -0.003 |  |
| a_invest | log1p(a_in3) | -0.018 |  |
| a_invest | a_op_in | -0.014 |  |
| a_invest | m_xfer_share | -0.015 |  |


## 5. Single-feature train group-fold AUROC

Y2 n=17,356 base 7.3%; Y3 stressed n=5,648 base 7.1%. Sign from the train side of each fold. Seed 20260918. Night quote: size 0.617 (replica 0.617), days 0.711 (replica 0.711), `c_salary_month` 0.671 (replica 0.671). Night Y3 GBM stays 0.762 / 0.752.

Y3 singles (train group-fold): `a_transfer` 0.566 vs size 0.617 (quote 0.617 CONFIRM, Δ -0.051) vs days 0.711 (night 0.711 CONFIRM) vs salary 0.671 (quote 0.671 CONFIRM) vs n_tx 0.703. |signed| 0.653 gross 0.664 has_xfer 0.655. `a_invest` 0.505 |invest| 0.554. Y2 transfer 0.524 vs size 0.552 / days 0.571; invest 0.488. Night Y3 quote stays 0.762 / 0.752 (not this cut).

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_transfer | 5,648 | 402 | 0.566 | 0.049 | -1 | 0.572 | 0.643 0.530 0.518 0.568 0.572 |
| y3_recover_cash_6m | abs_xfer | 5,648 | 402 | 0.653 | 0.061 | -1 | 0.654 | 0.755 0.641 0.616 0.596 0.655 |
| y3_recover_cash_6m | xfer_gross | 5,648 | 402 | 0.664 | 0.063 | -1 | 0.666 | 0.771 0.639 0.629 0.616 0.664 |
| y3_recover_cash_6m | has_xfer | 5,648 | 402 | 0.655 | 0.058 | -1 | 0.655 | 0.756 0.636 0.628 0.607 0.648 |
| y3_recover_cash_6m | washed | 5,648 | 402 | 0.523 | 0.014 | -1 | 0.525 | 0.542 0.505 0.516 0.532 0.522 |
| y3_recover_cash_6m | a_invest | 5,648 | 402 | 0.505 | 0.027 | 1 | 0.507 | 0.510 0.463 0.527 0.530 0.498 |
| y3_recover_cash_6m | abs_inv | 5,648 | 402 | 0.554 | 0.032 | -1 | 0.556 | 0.517 0.585 0.591 0.541 0.537 |
| y3_recover_cash_6m | inv_gross | 5,648 | 402 | 0.552 | 0.035 | -1 | 0.555 | 0.511 0.586 0.591 0.536 0.538 |
| y3_recover_cash_6m | has_inv | 5,648 | 402 | 0.552 | 0.034 | -1 | 0.554 | 0.512 0.586 0.590 0.535 0.536 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.714 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | c_salary_month | 5,648 | 402 | 0.671 | 0.050 | -1 | 0.672 | 0.611 0.659 0.745 0.689 0.652 |
| y3_recover_cash_6m | c_ss_month | 5,648 | 402 | 0.693 | 0.031 | -1 | 0.690 | 0.673 0.667 0.745 0.687 0.694 |
| y3_recover_cash_6m | m_xfer_share | 5,536 | 372 | 0.656 | 0.078 | -1 | 0.657 | 0.791 0.633 0.599 0.607 0.649 |
| y2_neg_2of3 | a_transfer | 17,356 | 1,271 | 0.524 | 0.047 | 1 | 0.540 | 0.570 0.463 0.547 0.486 0.553 |
| y2_neg_2of3 | abs_xfer | 17,356 | 1,271 | 0.551 | 0.069 | 1 | 0.573 | 0.662 0.502 0.508 0.507 0.573 |
| y2_neg_2of3 | xfer_gross | 17,356 | 1,271 | 0.562 | 0.073 | 1 | 0.588 | 0.676 0.500 0.518 0.524 0.592 |
| y2_neg_2of3 | has_xfer | 17,356 | 1,271 | 0.547 | 0.053 | 1 | 0.564 | 0.627 0.523 0.508 0.504 0.573 |
| y2_neg_2of3 | washed | 17,356 | 1,271 | 0.516 | 0.009 | 1 | 0.520 | 0.520 0.508 0.509 0.528 0.513 |
| y2_neg_2of3 | a_invest | 17,356 | 1,271 | 0.488 | 0.025 | -1 | 0.506 | 0.455 0.522 0.482 0.501 0.477 |
| y2_neg_2of3 | abs_inv | 17,356 | 1,271 | 0.479 | 0.016 | 1 | 0.501 | 0.504 0.473 0.461 0.473 0.486 |
| y2_neg_2of3 | inv_gross | 17,356 | 1,271 | 0.483 | 0.022 | 1 | 0.504 | 0.494 0.472 0.461 0.472 0.515 |
| y2_neg_2of3 | has_inv | 17,356 | 1,271 | 0.484 | 0.021 | 1 | 0.504 | 0.493 0.472 0.466 0.472 0.516 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.046 | 1 | 0.577 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 0.044 | 1 | 0.601 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | c_salary_month | 17,356 | 1,271 | 0.522 | 0.055 | 1 | 0.513 | 0.511 0.572 0.535 0.432 0.559 |
| y2_neg_2of3 | c_ss_month | 17,356 | 1,271 | 0.539 | 0.043 | -1 | 0.531 | 0.509 0.562 0.516 0.604 0.505 |
| y2_neg_2of3 | m_xfer_share | 16,764 | 1,236 | 0.563 | 0.076 | 1 | 0.589 | 0.678 0.499 0.512 0.521 0.604 |


KEEP-as-later-A: leftover after days/n_tx/salary beats size by ≥0.02 **and** not a twin **and** not SIZE **and** leftover after \|transfer\| vs signed **and** month shock. Still not on the 15-col card.

## 6. Residual AUROC after days / n_tx / salary

Honest leftover is `has_xfer` after days 0.579 / after card 0.564 (Δsize -0.053); |signed| after days 0.678. Signed OLS leftover after days 0.666 is a zero-month −days leak (ρ(resid,days)=-0.881; has-resid ρ vs days -0.089). Among transfer CM only, signed Y3 0.607 leftover-days 0.560. Invest signed leftover 0.686 / has_inv leftover 0.607. Leftover dies — CLOSE as quiet twin (explains SHAP-heavy / perm-light).

| y | feature | n | n_pos | CV | sign | Δsize | slope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | xfer_resid_days | 5,648 | 402 | 0.666 | 1 | 0.049 | 96865.352 |
| y3_recover_cash_6m | xfer_resid_ntx | 5,648 | 402 | 0.651 | 1 | 0.034 | 5970.325 |
| y3_recover_cash_6m | xfer_resid_sal | 5,648 | 402 | 0.629 | 1 | 0.013 | 2231087.945 |
| y3_recover_cash_6m | xfer_resid_ss | 5,648 | 402 | 0.648 | 1 | 0.031 | 1724197.264 |
| y3_recover_cash_6m | xfer_resid_days_ntx | 5,648 | 402 | 0.662 | 1 | 0.045 | 23116.739 |
| y3_recover_cash_6m | xfer_resid_days_sal | 5,648 | 402 | 0.699 | 1 | 0.083 | 55581.295 |
| y3_recover_cash_6m | xfer_resid_card | 5,648 | 402 | 0.668 | 1 | 0.052 | -15797.606 |
| y3_recover_cash_6m | xfer_resid_abs | 5,648 | 402 | 0.566 | -1 | -0.051 | 0.999 |
| y3_recover_cash_6m | xfer_resid_gross | 5,648 | 402 | 0.547 | -1 | -0.070 | 0.030 |
| y3_recover_cash_6m | has_resid_days | 5,648 | 402 | 0.579 | -1 | -0.038 | 0.024 |
| y3_recover_cash_6m | has_resid_ntx | 5,648 | 402 | 0.583 | -1 | -0.033 | 0.001 |
| y3_recover_cash_6m | has_resid_sal | 5,648 | 402 | 0.591 | -1 | -0.026 | 0.355 |
| y3_recover_cash_6m | has_resid_card | 5,648 | 402 | 0.564 | -1 | -0.053 | 0.018 |
| y3_recover_cash_6m | abs_resid_days | 5,648 | 402 | 0.678 | 1 | 0.062 | 103548.327 |
| y3_recover_cash_6m | abs_resid_card | 5,648 | 402 | 0.680 | 1 | 0.063 | -13562.594 |
| y3_recover_cash_6m | gros_resid_days | 5,648 | 402 | 0.678 | 1 | 0.061 | 464566.641 |
| y3_recover_cash_6m | inv_resid_days | 5,648 | 402 | 0.686 | -1 | 0.070 | -530.435 |
| y3_recover_cash_6m | inv_resid_ntx | 5,648 | 402 | 0.670 | -1 | 0.054 | -17.109 |
| y3_recover_cash_6m | inv_resid_card | 5,648 | 402 | 0.462 | 1 | -0.155 | -790.778 |
| y3_recover_cash_6m | has_inv_resid_days | 5,648 | 402 | 0.607 | 1 | -0.009 | 0.010 |
| y2_neg_2of3 | xfer_resid_days | 17,356 | 1,271 | 0.551 | -1 | -0.000 | 96865.352 |
| y2_neg_2of3 | xfer_resid_ntx | 17,356 | 1,271 | 0.569 | -1 | 0.017 | 5970.325 |
| y2_neg_2of3 | xfer_resid_sal | 17,356 | 1,271 | 0.465 | 1 | -0.087 | 2231087.945 |
| y2_neg_2of3 | xfer_resid_ss | 17,356 | 1,271 | 0.554 | 1 | 0.002 | 1724197.264 |
| y2_neg_2of3 | xfer_resid_days_ntx | 17,356 | 1,271 | 0.566 | -1 | 0.014 | 23116.739 |
| y2_neg_2of3 | xfer_resid_days_sal | 17,356 | 1,271 | 0.539 | -1 | -0.013 | 55581.295 |
| y2_neg_2of3 | xfer_resid_card | 17,356 | 1,271 | 0.562 | -1 | 0.010 | -15797.606 |
| y2_neg_2of3 | xfer_resid_abs | 17,356 | 1,271 | 0.524 | 1 | -0.028 | 0.999 |
| y2_neg_2of3 | xfer_resid_gross | 17,356 | 1,271 | 0.508 | 1 | -0.043 | 0.030 |
| y2_neg_2of3 | has_resid_days | 17,356 | 1,271 | 0.521 | 1 | -0.031 | 0.024 |
| y2_neg_2of3 | has_resid_ntx | 17,356 | 1,271 | 0.469 | 1 | -0.083 | 0.001 |
| y2_neg_2of3 | has_resid_sal | 17,356 | 1,271 | 0.545 | 1 | -0.007 | 0.355 |
| y2_neg_2of3 | has_resid_card | 17,356 | 1,271 | 0.524 | 1 | -0.028 | 0.018 |
| y2_neg_2of3 | abs_resid_days | 17,356 | 1,271 | 0.546 | -1 | -0.006 | 103548.327 |
| y2_neg_2of3 | abs_resid_card | 17,356 | 1,271 | 0.561 | -1 | 0.009 | -13562.594 |
| y2_neg_2of3 | gros_resid_days | 17,356 | 1,271 | 0.546 | -1 | -0.006 | 464566.641 |
| y2_neg_2of3 | inv_resid_days | 17,356 | 1,271 | 0.573 | 1 | 0.021 | -530.435 |
| y2_neg_2of3 | inv_resid_ntx | 17,356 | 1,271 | 0.586 | 1 | 0.035 | -17.109 |
| y2_neg_2of3 | inv_resid_card | 17,356 | 1,271 | 0.538 | 1 | -0.014 | -790.778 |
| y2_neg_2of3 | has_inv_resid_days | 17,356 | 1,271 | 0.565 | -1 | 0.013 | 0.010 |
| y3_recover_cash_6m | signed_on_xfer_cm | 3,016 | 99 | 0.607 | -1 | — | — |
| y3_recover_cash_6m | signed_on_xfer_resid_days | 3,016 | 99 | 0.560 | 1 | — | — |


## 7. ICC / company-demean (trait vs month shock)

`a_transfer` ICC=0.956 acf1=-0.028 (feature-report 0.96 BETWEEN / LOW_PERSIST). Y3 raw 0.566 vs company-demean 0.509 (drop 0.058). `has_xfer` ICC=0.958. `a_invest` ICC=0.384 acf1=-0.056. TRAIT (who transfers).

| col | ICC | acf1 | acf3 | Y3 raw | Y3 demean | drop | call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a_transfer | 0.956 | -0.028 | -0.050 | 0.566 | 0.509 | 0.058 | TRAIT |
| abs_xfer | 0.957 | -0.039 | -0.055 | 0.653 | 0.543 | 0.110 | TRAIT |
| has_xfer | 0.958 | 0.042 | -0.050 | 0.655 | 0.552 | 0.103 | TRAIT |
| xfer_gross | 0.956 | -0.023 | -0.050 | 0.664 | 0.540 | 0.124 | TRAIT |
| a_invest | 0.384 | -0.056 | -0.053 | 0.505 | 0.468 | 0.037 | SHOCK |
| has_inv | 0.949 | -0.048 | -0.062 | 0.552 | 0.528 | 0.024 | TRAIT |


## 8. Dark 470 vs invoiced 744

Train last-month: ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Mean company transfer-CM: invoiced 38.5% vs dark 41.5%. Same bank-book transfer rate. Y3 `a_transfer` invoiced 0.558 / dark 0.589. Holdout ever-ERP coverage only: 40/72.

| group | n_co | ever_xfer | xfer_cm | ever_inv | inv_cm |
| --- | --- | --- | --- | --- | --- |
| ever_erp_744 | 744 | 74.7% | 38.5% | 33.6% | 9.8% |
| never_erp_470 | 470 | 69.4% | 41.5% | 32.3% | 10.3% |


| group | n_cm | n_co | xfer | inv |
| --- | --- | --- | --- | --- |
| ever_erp | 13,554 | 744 | 37.4% | 10.1% |
| never_erp | 7,603 | 470 | 41.8% | 11.3% |


## 9. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)

Chronic 12 names (0158/0172, ≥50% labeled months below 0): 12. Y2 `a_transfer` 0.524 → drop-12 0.516. Transfer share on 12 92.9% vs rest 38.3%. Drop does not flip Y2 (≥0.03).

| y | feature | full | drop12 | Δ |
| --- | --- | --- | --- | --- |
| y2_neg_2of3 | a_transfer | 0.524 | 0.516 | 0.008 |
| y3_recover_cash_6m | a_transfer | 0.566 | 0.565 | 0.001 |
| y2_neg_2of3 | has_xfer | 0.547 | 0.529 | 0.018 |
| y3_recover_cash_6m | has_xfer | 0.655 | 0.651 | 0.003 |
| y2_neg_2of3 | a_invest | 0.488 | 0.481 | 0.006 |
| y3_recover_cash_6m | a_invest | 0.505 | 0.481 | 0.025 |
| y2_neg_2of3 | log1p_a_in3 | 0.552 | 0.545 | 0.007 |
| y3_recover_cash_6m | log1p_a_in3 | 0.617 | 0.617 | -0.000 |
| y2_neg_2of3 | c_n_days_with_tx | 0.571 | 0.549 | 0.022 |
| y3_recover_cash_6m | c_n_days_with_tx | 0.711 | 0.708 | 0.004 |


## 10. Q6 — lag1 / lag3 on short vs long books

Y3 contemporaneous 0.566 loses to size 0.617 / days 0.711 (lag1 0.557 / lag3 0.542). CLOSE as Q6 — no useful contemporaneous skill to lead. SHAP lag1 perm-light CONFIRMED.

| y | slice | col | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | all | a_transfer | 5,648 | 402 | 0.566 | -1 |
| y3_recover_cash_6m | all | a_transfer_lag1 | 5,648 | 402 | 0.557 | -1 |
| y3_recover_cash_6m | all | a_transfer_lag3 | 5,078 | 355 | 0.542 | -1 |
| y3_recover_cash_6m | all | a_invest | 5,648 | 402 | 0.505 | 1 |
| y3_recover_cash_6m | all | a_invest_lag1 | 5,648 | 402 | 0.506 | 1 |
| y3_recover_cash_6m | all | a_invest_lag3 | 5,078 | 355 | 0.474 | 1 |
| y3_recover_cash_6m | short | a_transfer | 146 | 13 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_transfer_lag1 | 146 | 13 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_transfer_lag3 | 53 | 1 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_invest | 146 | 13 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_invest_lag1 | 146 | 13 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_invest_lag3 | 53 | 1 | LOW_POWER | — |
| y3_recover_cash_6m | long | a_transfer | 5,502 | 389 | 0.570 | -1 |
| y3_recover_cash_6m | long | a_transfer_lag1 | 5,502 | 389 | 0.559 | -1 |
| y3_recover_cash_6m | long | a_transfer_lag3 | 5,025 | 354 | 0.544 | -1 |
| y3_recover_cash_6m | long | a_invest | 5,502 | 389 | 0.504 | 1 |
| y3_recover_cash_6m | long | a_invest_lag1 | 5,502 | 389 | 0.505 | 1 |
| y3_recover_cash_6m | long | a_invest_lag3 | 5,025 | 354 | 0.473 | 1 |
| y2_neg_2of3 | all | a_transfer | 17,356 | 1,271 | 0.524 | 1 |
| y2_neg_2of3 | all | a_transfer_lag1 | 16,161 | 1,151 | 0.517 | 1 |
| y2_neg_2of3 | all | a_transfer_lag3 | 13,776 | 938 | 0.526 | 1 |
| y2_neg_2of3 | all | a_invest | 17,356 | 1,271 | 0.488 | -1 |
| y2_neg_2of3 | all | a_invest_lag1 | 16,161 | 1,151 | 0.487 | -1 |
| y2_neg_2of3 | all | a_invest_lag3 | 13,776 | 938 | 0.484 | -1 |
| y2_neg_2of3 | short | a_transfer | 1,833 | 147 | 0.463 | -1 |
| y2_neg_2of3 | short | a_transfer_lag1 | 1,497 | 108 | 0.346 | -1 |
| y2_neg_2of3 | short | a_transfer_lag3 | 830 | 46 | LOW_POWER | — |
| y2_neg_2of3 | short | a_invest | 1,833 | 147 | 0.516 | 1 |
| y2_neg_2of3 | short | a_invest_lag1 | 1,497 | 108 | 0.515 | 1 |
| y2_neg_2of3 | short | a_invest_lag3 | 830 | 46 | LOW_POWER | — |
| y2_neg_2of3 | long | a_transfer | 15,523 | 1,124 | 0.537 | 1 |
| y2_neg_2of3 | long | a_transfer_lag1 | 14,664 | 1,043 | 0.528 | 1 |
| y2_neg_2of3 | long | a_transfer_lag3 | 12,946 | 892 | 0.534 | 1 |
| y2_neg_2of3 | long | a_invest | 15,523 | 1,124 | 0.489 | -1 |
| y2_neg_2of3 | long | a_invest_lag1 | 14,664 | 1,043 | 0.487 | -1 |
| y2_neg_2of3 | long | a_invest_lag3 | 12,946 | 892 | 0.482 | -1 |


## 11. Invest — deploy vs return (G rise-only?)

Invest CM: deploy-only 920, return-only 923, both 388 / any 2,231. Deploy txs-months 1,308 (signed p50 -4,964); return 1,311 (p50 1,590). |net|/gross p50 among invest CM 1.000. NO — both deploy and return fire (not a G-style rise-only connection).

| slice | n_cm | share_of_any | signed p50 |
| --- | --- | --- | --- |
| deploy only | 920 | 41.2% | -4,319 |
| return only | 923 | 41.4% | 814 |
| both | 388 | 17.4% | 0 |


| y | slice | n_cm | n_lab | n_pos | rate |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | deploy_only | 920 | 770 | 58 | 7.5% |
| y2_neg_2of3 | return_only | 923 | 768 | 44 | 5.7% |
| y2_neg_2of3 | both | 388 | 313 | 42 | 13.4% |
| y2_neg_2of3 | no_invest | 18,926 | 15,505 | 1,127 | 7.3% |
| y3_recover_cash_6m | deploy_only | 920 | 333 | 6 | 1.8% |
| y3_recover_cash_6m | return_only | 923 | 375 | 6 | 1.6% |
| y3_recover_cash_6m | both | 388 | 140 | 8 | 5.7% |
| y3_recover_cash_6m | no_invest | 18,926 | 4,800 | 382 | 8.0% |


## 12. Amounts — 1€ token or real mass?

Train transfer |amt| p50=6,000 p90=500,000 (≤1€ 0.7%) vs op_out p50=497 (ratio 12.062). Real mass — transfer tickets are larger than typical op_out.

| token | n_tx | p50 |amt| | p90 |amt| | ≤1€ |
| --- | --- | --- | --- | --- |
| transfer | 146,071 | 6,000 | 500,000 | 0.7% |
| op_out | 769,371 | 497 | 28,500 | 2.1% |
| investment_deployment | 3,715 | 1,308 | 58,782 | 6.4% |
| investment_return | 3,214 | 594 | 60,000 | 4.9% |


## 13. Category token mix / descriptions (Q5 leftover?)

Transfer txs n=146,071; missing counterparty_id 97.8% (cannot score intercompany via CP — `d_interco_share` is already all-null). Description mix: TRASP 86.4%, TRANSFERENCIA 7.6%, owner/sweep/dividend 0.0%. No KEEP-Q5 footnote — this is bookkeeping TRASPASO / TRANSFERENCIA, already the quiet-month story.

| token | share | n_tx |
| --- | --- | --- |
| TRASP / TRASPASO | 86.4% | 126,273 |
| TRANSFERENCIA | 7.6% | 11,110 |
| COUNTERPARTY_ | 6.9% | 10,051 |
| [ACCOUNT] | 36.7% | 53,553 |
| [COMPANY] | 15.8% | 23,112 |
| [NUM] | 29.7% | 43,327 |
| SWEEP / TREAS / OWNER | 0.0% | 30 |
| INTER / GRUPO | 1.5% | 2,119 |


## 14. Y rates on has / wash / one-way

Y3 rate no-transfer 11.5% vs has-transfer 3.3% (gap 8.2%). Quiet months recover — same sign as salary / days / n_tx.

| y | slice | n_cm | n_lab | n_pos | rate |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | has_xfer | 8,242 | 6,698 | 642 | 9.6% |
| y2_neg_2of3 | no_xfer | 12,915 | 10,658 | 629 | 5.9% |
| y2_neg_2of3 | one_way | 4,574 | 3,686 | 255 | 6.9% |
| y2_neg_2of3 | both_ways | 3,955 | 3,242 | 421 | 13.0% |
| y2_neg_2of3 | washed | 777 | 664 | 96 | 14.5% |
| y2_neg_2of3 | has_inv | 2,231 | 1,851 | 144 | 7.8% |
| y2_neg_2of3 | no_inv | 18,926 | 15,505 | 1,127 | 7.3% |
| y3_recover_cash_6m | has_xfer | 8,242 | 3,016 | 99 | 3.3% |
| y3_recover_cash_6m | no_xfer | 12,915 | 2,632 | 303 | 11.5% |
| y3_recover_cash_6m | one_way | 4,574 | 1,470 | 70 | 4.8% |
| y3_recover_cash_6m | both_ways | 3,955 | 1,660 | 30 | 1.8% |
| y3_recover_cash_6m | washed | 777 | 342 | 6 | 1.8% |
| y3_recover_cash_6m | has_inv | 2,231 | 848 | 20 | 2.4% |
| y3_recover_cash_6m | no_inv | 18,926 | 4,800 | 382 | 8.0% |


## 15. Size / days terciles + busy-half

Y3 `a_transfer` inside size T1/T2/T3: 0.538 / 0.543 / 0.629. Inside days terciles: 0.545 / 0.443 / 0.666. Busy-half days CV 0.600; quiet-half 0.521.

| clock | tercile | n | n_pos | xfer | CV |
| --- | --- | --- | --- | --- | --- |
| size_t | T1 | 1,572 | 269 | 38.8% | 0.538 |
| size_t | T2 | 2,044 | 73 | 53.0% | 0.543 |
| size_t | T3 | 2,032 | 60 | 65.1% | 0.629 |
| days_t | T1 | 1,575 | 271 | 33.3% | 0.545 |
| days_t | T2 | 2,112 | 79 | 51.5% | 0.443 |
| days_t | T3 | 1,961 | 52 | 71.6% | 0.666 |


## 16. Holdout coverage only (no AUROC)

Holdout 72 coverage only: 1,073 CM. has_xfer 31.9%; has_inv 14.3%. No AUROC claim.

| col | n_cm | n_co | cov | mean | p50 |
| --- | --- | --- | --- | --- | --- |
| a_transfer | 1073 | 72 | 100.0% | -42955.395 | 0.000 |
| a_invest | 1073 | 72 | 100.0% | -2475.374 | 0.000 |
| has_xfer | 1073 | 72 | 100.0% | 0.319 | 0.000 |
| has_inv | 1073 | 72 | 100.0% | 0.143 | 0.000 |
| xfer_gross | 1073 | 72 | 100.0% | 294656.757 | 0.000 |


## 17. Calendar

Transfer-CM share range 33.6%–42.4%. Flat — not a tax-style calendar dummy.

| month | n_cm | xfer | inv | signed p50 |
| --- | --- | --- | --- | --- |
| Jan | 1,768 | 38.7% | 10.1% | 0 |
| Feb | 1,881 | 39.1% | 9.9% | 0 |
| Mar | 1,925 | 40.6% | 10.2% | 0 |
| Apr | 1,945 | 39.9% | 10.7% | 0 |
| May | 1,966 | 40.8% | 10.9% | 0 |
| Jun | 1,976 | 41.4% | 11.5% | 0 |
| Jul | 2,010 | 42.4% | 11.9% | 0 |
| Aug | 2,047 | 36.8% | 8.7% | 0 |
| Sep | 1,299 | 36.0% | 10.7% | 0 |
| Oct | 1,381 | 35.0% | 11.3% | 0 |
| Nov | 1,428 | 33.6% | 10.4% | 0 |
| Dec | 1,531 | 39.6% | 10.5% | 0 |


## 18. Signed leftover after the presence flag

Y3 has_xfer 0.655; signed leftover after has_xfer 0.618; after has+days 0.691. Signed euros keep some leftover after presence.

| y | feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_transfer | 5,648 | 402 | 0.566 | -1 |
| y3_recover_cash_6m | has_xfer | 5,648 | 402 | 0.655 | -1 |
| y3_recover_cash_6m | resid_has | 5,648 | 402 | 0.618 | 1 |
| y3_recover_cash_6m | resid_has_days | 5,648 | 402 | 0.691 | 1 |
| y2_neg_2of3 | a_transfer | 17,356 | 1,271 | 0.524 | 1 |
| y2_neg_2of3 | has_xfer | 17,356 | 1,271 | 0.547 | 1 |
| y2_neg_2of3 | resid_has | 17,356 | 1,271 | 0.532 | -1 |
| y2_neg_2of3 | resid_has_days | 17,356 | 1,271 | 0.558 | -1 |


## 19. Ever-transfer company trait

Ever-transfer companies 882/1214. Y3 ever-flag 0.572; company mean has_xfer 0.747. A company trait that a transferer is less likely to recover — not a month shock.

| y | feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | ever_xfer | 17,356 | 1,271 | 0.485 | -1 |
| y2_neg_2of3 | mean_has_xfer | 17,356 | 1,271 | 0.554 | 1 |
| y2_neg_2of3 | ever_inv | 17,356 | 1,271 | 0.480 | -1 |
| y3_recover_cash_6m | ever_xfer | 5,648 | 402 | 0.572 | -1 |
| y3_recover_cash_6m | mean_has_xfer | 5,648 | 402 | 0.747 | -1 |
| y3_recover_cash_6m | ever_inv | 5,648 | 402 | 0.618 | -1 |


## 20. Winsor signed + leftover on busy/quiet halves

Winsor p01–p99 signed Y3 0.566; resid-days 0.608 (still the zero-month leak if high). has_xfer on busy-half 0.615; quiet-half 0.586.

| feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- |
| winsor_signed | 5,648 | 402 | 0.566 | -1 |
| winsor_resid_days | 5,648 | 402 | 0.608 | 1 |
| winsor_resid_card | 5,648 | 402 | 0.629 | 1 |
| has_xfer_busy | 3,600 | 131 | 0.615 | -1 |
| has_xfer_quiet | 2,048 | 271 | 0.586 | -1 |
| abs_xfer_busy | 3,600 | 131 | 0.614 | -1 |


## 21. Company-mean has_xfer vs activity (the 0.747 trait)

Company-mean has_xfer vs mean days ρ=0.518, n_tx 0.495, salary 0.523, size 0.426. Company-level Y3-any AUROC mean_has 0.743 vs mean days 0.706 vs size 0.730. Not a |ρ|≥0.80 twin of company-mean days, but ICC 0.96 still makes it a trait, not a month shock.

| vs | ρ | flag |
| --- | --- | --- |
| days | 0.518 |  |
| n_tx | 0.495 |  |
| salary | 0.523 |  |
| ss | 0.506 |  |
| size | 0.426 |  |
| Y3-any company AUROC mean_has | 0.743 | days 0.7057915611003296 size 0.7303626824712668 |


## 22. |signed| leftover on transfer months only

Among transfer CM: |signed| 0.520 gross 0.569; leftover after days 0.629 / card 0.598. Euro mass inside transfer months is not a leftover lever.

| feature | n | n_pos | CV |
| --- | --- | --- | --- |
| abs_on_xfer | 3,016 | 99 | 0.520 |
| gros_on_xfer | 3,016 | 99 | 0.569 |
| abs_on_resid_days | 3,016 | 99 | 0.629 |
| gros_on_resid_days | 3,016 | 99 | 0.614 |
| abs_on_resid_card | 3,016 | 99 | 0.598 |


## What failed / next (held for wave note)

- a_transfer Y3 0.566 has_xfer leftover-card 0.564 < size+0.02 or trait/twin/wash/SIZE — CLOSE as X

Elapsed 11s. Cuts: prevalence, tokens, wash, Spearman, singles, honest leftover (has_xfer after days — signed OLS leftover is a zero-month −days leak), ICC, dark 470/744, chronic-12, Q6 lags, invest sidedness, amounts, description mix, Y rates, terciles, holdout, calendar, leftover after has_xfer, ever-transfer trait, winsor / busy-half, company-mean activity twin, |signed| leftover on transfer months only.
