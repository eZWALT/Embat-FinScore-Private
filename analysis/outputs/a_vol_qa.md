# a_vol — in-memory Javier cashflow volatility

- **When:** 2026-09-19T04:17:23+02:00
- **Agent:** `6b456387`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.evaluate.a_vol_qa`
- **Holdout:** 72 companies, seed 20260918. Coverage only. ρ / singles on train.
- **X:** `a_vol` reconstructed in memory = `min(3, sd6(a_net) / max(mean6(a_op_in), 1))`. Family A — legal for Y2/Y3.
- **Never B as X.** `b_bal_vol` is a diagnostic only.
- **Y:** `y3_recover_cash_6m` (stressed-only labeled) and `y2_neg_2of3`.
- Not a 0–100. Not a parquet column. Not a 15-col retrain.

## Decision

**CLOSE** as a store candidate. **PARK** as a health Y. Merge recommendation: NO — Y3 loses to size (Δ=+0.009); Y2 loses to size (Δ=-0.112). a_out_vol later-store KEEP=False (trait dummy, not month shock).

- Identity vs Javier raw `volatility`: ρ=1.000 n=15,089 **SAME** (Pearson=1.000 max|Δ|=0 exact=100.0%).
- vs `b_bal_vol`: ρ=0.354 n=14,968 **DRIFT** (Javier twin quote 0.354).
- SIZE ρ vs `log1p(a_in3)`: -0.391 (ok).
- Copy: vs `a_io_ratio` ρ=-0.333; vs `a_growth_3` ρ=-0.121 (not a twin).
- Y3 stressed single: `0.626` vs size `0.617` (Δ +0.009) vs `c_n_days_with_tx` night 0.711 (replica 0.711).
- Y2 single: `0.439` vs size `0.551` (Δ -0.112) vs night GBM 0.540. Fold-min 0.377 invert.
- Q6 lag1 KEEP: **False**.
- Chronic-12 drop does not flip the gate (see cut 8).
- `a_out_vol` (in memory, not Javier): Y3 0.722 Δ+0.104, T2+T3 0.726 Δ+0.167; Y2 0.601 Δ+0.049 fold-min 0.443 invert. Quiet-days +9.8pp / dark +11.5pp / ERP +8.3pp. Not SIZE/COPY/B/15-col. Y3-only. CLOSE as trait dummy. Not tonight's card.
- Javier `a_vol` loses to size inside T1 (Δ -0.028) and inside T2+T3 (Δ -0.027). Overall +0.009 is between-tercile (small books are high-vol and have a higher Y3 base). CLOSE.
- `a_out_vol` is a company trait (demean drops 0.722→0.549), not a days twin (company ρ=-0.453), complementary 2×2 lifts +13.2% quiet / +13.1% busy. Q5 overlap with `a_vol` is 51.1% (not same tail); out-only recoveries are 58.0% T2+T3. Not a 15-col lag card. CLOSE / PARK / merge NO.

## Parent return

| question | number |
| --- | ---: |
| ρ vs Javier `volatility` | 1.000 **SAME** |
| ρ vs `b_bal_vol` | 0.354 **DRIFT** |
| Y3 `a_vol` / size / days | 0.626 / 0.617 / 0.711 |
| Y2 `a_vol` / size / night | 0.439 / 0.551 / 0.540 |
| Merge `a_vol` | **NO** (CLOSE / PARK) |
| Q6 lag1 `a_vol` | **False** (Y3 lag1=0.620) |
| `a_out_vol` Y3 (in memory) | 0.722 Δ+0.104 folds 0.710 0.724 0.782 0.664 0.728 |
| `a_out_vol` drop a_vol Q5 | 0.720 vs size 0.453 (Δ +0.266) |
| Store already has a_vol? | no |

## Resume return — a_out_vol 0.722

| question | number |
| --- | ---: |
| Reproduced 0.722? | **True** (0.722) |
| Leak twin / SIZE | False / False |
| Company ρ vs days (Y3-labeled) | -0.451 |
| Drop 12 Δ | +0.002 (move=False) |
| Short <12 / long ≥12 | 0.742 / 0.724 |
| Q6 lag1 | 0.699 lift -0.022 |
| Month / company quiet 2×2 | +0.098 / +0.132 |
| Hot cell is 12 names? | month=False company=False |
| Demean / η² | 0.722→0.549 / 0.741 |
| Within-co residual lift | -0.012 |
| Trait vs shock | **trait** (cannot transfer as a month shock) |
| Leave-one-fold min mean | 0.706 still≥0.70=True |
| Company-OOF (group holdout) | 0.682 transfers=True |
| Company-OOF days / size | 0.666 / 0.631 |
| Pearson leak twin / SIZE | False / False |
| Max abs Spearman leak | -0.411 vs `a_n_tx` |
| Drop company-Q5 Δ | +0.066 (move=True) |
| Q1–Q4 company-OOF / body trait | 0.604 / False |
| Q5-only month CV | 0.540 inside=False |
| Q5 dummy vs continuous | 0.647 vs 0.722 |
| Holdout 72 a_out_vol / Y3 pos | 66.5% / 14 |
| Holdout at/above train Q5 | 21/71 (29.6%) |
| Company-mean perm p | 0.0010 (obs 0.688) |
| Company-mean AUROC bootstrap | 0.634–0.741 |
| Javier a_vol still CLOSE | True (Δ +0.009) |
| later-store KEEP | **False** (needs month shock) |
| a_out_vol X / Y / merge | **CLOSE** / **PARK** / **NO** |

## Brief questions

- Q3 turning: Javier named this `volatility`. Cashflow sd, not balance sd.
- Q5 why: only if the single beats size and is not a caja twin.
- Q6: honest 1-month lag only.

## 1. Identity vs Javier

Formula matches `score_pipeline.py` ~165: `min(3, sd6(net) / max(mean6(op_in), 1))`. Store reconstruct from A is identity.

| pair | Spearman | Pearson | max\|Δ\| | p50\|Δ\| | exact | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `volatility` vs `a_vol` | 1.000 | 1.000 | 0 | 0 | 100.0% | 15,089 |

Clip=3 share on defined train CM: 16.6%. Of all train CM that is the 11.8% pipeline quote (cut 11).

## 2. vs `b_bal_vol` (different object)

- `a_vol` vs `b_bal_vol`: ρ=0.354 n=14,968 **DRIFT**.
- Javier vs `b_bal_vol`: ρ=0.354 n=14,968.
- Per-company (n≥8): 802 cos; p10=-0.568 p50=0.232 p90=0.785; SAME 0.5% / CLOSE 8.4% / DRIFT 91.1%.

Do not use `b_bal_vol` as Y2/Y3 X.

## 3. SIZE + copy screen

| col | vs | ρ | n | |ρ|≥gate |
| --- | --- | --- | --- | --- |
| a_vol | size | -0.391 | 15,089 | ok |
| a_vol | a_io_ratio | -0.333 | 15,089 | ok |
| a_vol | a_growth_3 | -0.121 | 13,532 | ok |
| a_vol | a_net_margin | -0.335 | 15,089 | ok |
| a_in_vol | size | -0.163 | 15,089 | ok |
| a_in_vol | a_io_ratio | -0.016 | 15,089 | ok |
| a_in_vol | a_growth_3 | -0.164 | 13,532 | ok |
| a_in_vol | a_net_margin | -0.037 | 15,089 | ok |
| a_io_vol | size | -0.229 | 12,674 | ok |
| a_io_vol | a_io_ratio | -0.249 | 12,674 | ok |
| a_io_vol | a_growth_3 | -0.117 | 11,353 | ok |
| a_io_vol | a_net_margin | -0.272 | 12,674 | ok |
| a_io_sd6 | size | -0.075 | 12,674 | ok |
| a_io_sd6 | a_io_ratio | 0.136 | 12,674 | ok |
| a_io_sd6 | a_growth_3 | -0.079 | 11,353 | ok |
| a_io_sd6 | a_net_margin | 0.117 | 12,674 | ok |
| a_vol_sumdenom | size | -0.400 | 15,089 | ok |
| a_vol_sumdenom | a_io_ratio | -0.340 | 15,089 | ok |
| a_vol_sumdenom | a_growth_3 | -0.122 | 13,532 | ok |
| a_vol_sumdenom | a_net_margin | -0.339 | 15,089 | ok |
| b_bal_vol | size | -0.280 | 14,968 | ok |
| b_bal_vol | a_io_ratio | 0.189 | 14,968 | ok |
| b_bal_vol | a_growth_3 | -0.014 | 13,421 | ok |
| b_bal_vol | a_net_margin | 0.211 | 14,968 | ok |


## 4. Single-feature train group-fold

Y2 n=17,356 base 7.3%; Y3 stressed n=5,648 base 7.1%. Sign from the train side of each fold. Seed 20260918.

### Y3 (stressed)

| feature | CV ± sd | train | sign | cov | gap vs size | gap vs 0.711 | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a_vol` | 0.626 ± 0.062 | 0.620 | +1 | 0.746 | +0.009 | -0.085 | 0.552 0.715 0.656 0.594 0.614 |
| `a_in_vol` | 0.635 ± 0.040 | 0.635 | +1 | 0.746 | +0.017 | -0.076 | 0.676 0.587 0.637 0.602 0.672 |
| `a_io_vol` | 0.574 ± 0.024 | 0.568 | +1 | 0.603 | -0.044 | -0.137 | 0.577 0.613 0.563 0.566 0.550 |
| `a_io_sd6` | 0.562 ± 0.075 | 0.556 | +1 | 0.603 | -0.055 | -0.149 | 0.676 0.530 0.496 0.597 0.511 |
| `log1p_a_in3` | 0.617 ± 0.060 | 0.620 | -1 | 0.979 | +0.000 | -0.094 | 0.565 0.632 0.683 0.544 0.661 |
| `c_n_days_with_tx` | 0.711 ± 0.031 | 0.723 | -1 | 1.000 | +0.094 | +0.000 | 0.665 0.738 0.700 0.715 0.740 |
| `a_io_ratio` | 0.565 ± 0.043 | 0.563 | -1 | 0.979 | -0.052 | -0.146 | 0.523 0.578 0.632 0.550 0.541 |
| `a_growth_3` | 0.515 ± 0.046 | 0.510 | -1 | 0.709 | -0.103 | -0.196 | 0.573 0.528 0.518 0.445 0.511 |


### Y2

| feature | CV ± sd | train | sign | cov | gap vs size | gap vs 0.540 | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a_vol` | 0.439 ± 0.044 | 0.506 | +1 | 0.661 | -0.112 | -0.101 | 0.429 0.377 0.488 0.474 0.427 |
| `a_in_vol` | 0.531 ± 0.049 | 0.515 | -1 | 0.661 | -0.020 | -0.009 | 0.481 0.596 0.569 0.494 0.516 |
| `a_io_vol` | 0.571 ± 0.069 | 0.538 | -1 | 0.550 | +0.020 | +0.031 | 0.464 0.650 0.593 0.596 0.554 |
| `a_io_sd6` | 0.571 ± 0.041 | 0.547 | -1 | 0.550 | +0.020 | +0.031 | 0.546 0.606 0.625 0.538 0.539 |
| `log1p_a_in3` | 0.551 ± 0.046 | 0.540 | +1 | 0.862 | +0.000 | +0.011 | 0.523 0.609 0.594 0.519 0.511 |
| `c_n_days_with_tx` | 0.571 ± 0.046 | 0.577 | +1 | 1.000 | +0.020 | +0.031 | 0.623 0.539 0.612 0.565 0.517 |
| `a_io_ratio` | 0.473 ± 0.069 | 0.509 | +1 | 0.862 | -0.079 | -0.067 | 0.438 0.571 0.481 0.384 0.490 |
| `a_growth_3` | 0.516 ± 0.045 | 0.514 | +1 | 0.594 | -0.036 | -0.024 | 0.523 0.488 0.583 0.522 0.462 |


## 5. Quintiles

### `a_vol` vs `y3_recover_cash_6m` — n=4,212 bins=5 shape=**tail** mono↑=False tail=True

| Q | interval | n | n_pos | P(Y=1) | median X |
| --- | --- | --- | --- | --- | --- |
| 1 | (-0.001, 0.248] | 843 | 39 | 0.046 | 0.146 |
| 2 | (0.248, 0.466] | 842 | 49 | 0.058 | 0.3494 |
| 3 | (0.466, 0.826] | 842 | 44 | 0.052 | 0.612 |
| 4 | (0.826, 1.842] | 842 | 69 | 0.082 | 1.147 |
| 5 | (1.842, 3.0] | 843 | 112 | 0.133 | 3 |


### `a_in_vol` vs `y3_recover_cash_6m` — n=4,212 bins=5 shape=**tail** mono↑=False tail=True

| Q | interval | n | n_pos | P(Y=1) | median X |
| --- | --- | --- | --- | --- | --- |
| 1 | (-0.001, 0.305] | 843 | 51 | 0.060 | 0.1996 |
| 2 | (0.305, 0.536] | 842 | 26 | 0.031 | 0.4214 |
| 3 | (0.536, 0.842] | 842 | 45 | 0.053 | 0.6619 |
| 4 | (0.842, 1.386] | 842 | 64 | 0.076 | 1.066 |
| 5 | (1.386, 3.0] | 843 | 127 | 0.151 | 1.885 |


### `a_io_vol` vs `y3_recover_cash_6m` — n=3,408 bins=5 shape=**tail** mono↑=False tail=True

| Q | interval | n | n_pos | P(Y=1) | median X |
| --- | --- | --- | --- | --- | --- |
| 1 | (-0.001, 0.0969] | 682 | 53 | 0.078 | 0.04785 |
| 2 | (0.0969, 0.206] | 681 | 34 | 0.050 | 0.1489 |
| 3 | (0.206, 0.366] | 682 | 35 | 0.051 | 0.2719 |
| 4 | (0.366, 0.678] | 681 | 50 | 0.073 | 0.4898 |
| 5 | (0.678, 3.0] | 682 | 85 | 0.125 | 1.022 |


### `log1p_a_in3` vs `y3_recover_cash_6m` — n=5,528 bins=5 shape=**head** mono↑=False tail=False

| Q | interval | n | n_pos | P(Y=1) | median X |
| --- | --- | --- | --- | --- | --- |
| 1 | (-0.001, 11.204] | 1106 | 167 | 0.151 | 9.422 |
| 2 | (11.204, 12.737] | 1105 | 64 | 0.058 | 12.1 |
| 3 | (12.737, 13.728] | 1106 | 38 | 0.034 | 13.23 |
| 4 | (13.728, 14.707] | 1105 | 42 | 0.038 | 14.21 |
| 5 | (14.707, 22.706] | 1106 | 80 | 0.072 | 15.45 |


### `a_vol` vs `y2_neg_2of3` — n=11,477 bins=5 shape=**flat** mono↑=False tail=False

| Q | interval | n | n_pos | P(Y=1) | median X |
| --- | --- | --- | --- | --- | --- |
| 1 | (-0.001, 0.329] | 2296 | 130 | 0.057 | 0.1901 |
| 2 | (0.329, 0.624] | 2295 | 162 | 0.071 | 0.4651 |
| 3 | (0.624, 1.107] | 2295 | 160 | 0.070 | 0.8335 |
| 4 | (1.107, 2.449] | 2295 | 155 | 0.068 | 1.542 |
| 5 | (2.449, 3.0] | 2296 | 141 | 0.061 | 3 |


### `a_in_vol` vs `y2_neg_2of3` — n=11,477 bins=5 shape=**flat** mono↑=False tail=False

| Q | interval | n | n_pos | P(Y=1) | median X |
| --- | --- | --- | --- | --- | --- |
| 1 | (-0.001, 0.309] | 2296 | 138 | 0.060 | 0.1541 |
| 2 | (0.309, 0.586] | 2295 | 171 | 0.075 | 0.4531 |
| 3 | (0.586, 0.94] | 2295 | 170 | 0.074 | 0.7479 |
| 4 | (0.94, 1.549] | 2313 | 156 | 0.067 | 1.179 |
| 5 | (1.549, 3.0] | 2278 | 113 | 0.050 | 2.139 |


### `a_io_vol` vs `y2_neg_2of3` — n=9,551 bins=5 shape=**monotone_down** mono↑=False tail=False

| Q | interval | n | n_pos | P(Y=1) | median X |
| --- | --- | --- | --- | --- | --- |
| 1 | (-0.001, 0.101] | 1911 | 139 | 0.073 | 0.007189 |
| 2 | (0.101, 0.246] | 1910 | 131 | 0.069 | 0.1687 |
| 3 | (0.246, 0.443] | 1910 | 127 | 0.066 | 0.3329 |
| 4 | (0.443, 0.791] | 1910 | 110 | 0.058 | 0.5791 |
| 5 | (0.791, 3.0] | 1910 | 97 | 0.051 | 1.114 |


### `log1p_a_in3` vs `y2_neg_2of3` — n=14,968 bins=5 shape=**flat** mono↑=False tail=False

| Q | interval | n | n_pos | P(Y=1) | median X |
| --- | --- | --- | --- | --- | --- |
| 1 | (-0.001, 9.689] | 2994 | 160 | 0.053 | 0.004975 |
| 2 | (9.689, 11.859] | 2993 | 185 | 0.062 | 10.98 |
| 3 | (11.859, 13.133] | 2994 | 225 | 0.075 | 12.51 |
| 4 | (13.133, 14.449] | 2993 | 244 | 0.082 | 13.77 |
| 5 | (14.449, 22.713] | 2994 | 230 | 0.077 | 15.3 |


## 6. Q6 lag1 (honest 1-month)

Lag-1 KEEP needs beat size ≥0.02, stay <0.60, and beat lag-0 by ≥0.01.

| Y | feature | lag0 | lag1 | lift | gap vs size | Q6 KEEP |
| --- | --- | --- | --- | --- | --- | --- |
| y2 | `a_vol` | 0.439 | 0.440 | +0.001 | -0.111 | False |
| y2 | `a_in_vol` | 0.531 | 0.532 | +0.001 | -0.020 | False |
| y2 | `a_io_vol` | 0.571 | 0.565 | -0.007 | +0.013 | False |
| y3 | `a_vol` | 0.626 | 0.620 | -0.006 | +0.003 | False |
| y3 | `a_in_vol` | 0.635 | 0.619 | -0.016 | +0.001 | False |
| y3 | `a_io_vol` | 0.574 | 0.541 | -0.032 | -0.076 | False |


## 7. Alternatives (still in memory)

- `a_in_vol` = min(3, sd6(`a_op_in`) / max(mean6(`a_op_in`), 1))
- `a_io_vol` = min(3, sd6(`a_io_ratio`) / max(|mean6(io)|, 1e-6))
- `a_io_sd6` = sd6(`a_io_ratio`) unscaled
- `a_vol_sumdenom` = min(3, sd6(net) / max(sum6(`a_op_in`), 1)) — wrong denom check

| col | vs Javier | vs b_bal_vol | Y3 CV | Y3 Δsize | Y2 CV | Y2 Δsize | ρ size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a_vol` | 1.000 SAME | 0.354 | 0.626 | +0.009 | 0.439 | -0.112 | -0.391 |
| `a_in_vol` | 0.473 DRIFT | 0.206 | 0.635 | +0.017 | 0.531 | -0.020 | -0.163 |
| `a_io_vol` | 0.597 DRIFT | 0.137 | 0.574 | -0.044 | 0.571 | +0.020 | -0.229 |
| `a_io_sd6` | 0.303 DRIFT | 0.235 | 0.562 | -0.055 | 0.571 | +0.020 | -0.075 |
| `a_vol_sumdenom` | 0.998 SAME | 0.355 | — | — | — | — | -0.400 |


## 8. Chronic 12 dark Y2 names

Reconstructed from GROUP_0158/0172 with ≥50% labeled months `b_below_0` (y2_why.md). n=12. B used only to *name* the pile — never as X.

| slice | n | n_pos | med a_vol | clip3 |
| --- | --- | --- | --- | --- |
| train_y2 | 17356 | 1271 | 0.833 | 16.6% |
| chronic_12_y2 | 216 | 177 | 0.893 | 37.2% |
| rest_y2 | 17140 | 1094 | 0.831 | 16.3% |
| train_y3 | 5648 | 402 | 0.612 | 13.3% |
| chronic_12_y3 | 162 | 0 | 0.853 | 33.3% |
| rest_y3 | 5486 | 402 | 0.606 | 12.8% |


Singles after dropping the 12 names:

| Y | n | n_pos | a_vol CV | size CV | days CV | gap |
| --- | --- | --- | --- | --- | --- | --- |
| y3 | 5486 | 402 | 0.628 ± 0.059 | 0.618 | 0.708 | +0.011 |
| y2 | 17140 | 1094 | 0.495 ± 0.079 | 0.544 | 0.549 | -0.049 |


## 9. Holdout coverage (LOW_POWER)

Holdout CM=1,073 cos=72. `a_vol` finite 66.5%. Y2 pos=23 Y3 pos=14. Not a KEEP claim.

## 10. Train coverage + fold identity

Train CM=21,157 cos=1214. `a_vol` 71.3%; `a_in_vol` 71.3%; `a_io_vol` 59.9%. Fold-min ρ vs Javier 1.000.

## 11. Clip=3 quote (11.8% of all train CM)

Defined-row clip share 16.6% = 11.8% of all train CM. Javier of-all 11.8%. Confirm 11.8%: **True**.

## 12. Alternatives after drop 12 chronic

`a_io_vol` Y2 gap after drop12 = +0.019 (was +0.020). Is-the-12: **True**. days gap +0.005. `a_in_vol` Y3 drop12 CV=0.633 gap=+0.015.

| Y | feature | CV ± sd | gap vs size | folds |
| --- | --- | --- | --- | --- |
| y2 | `a_vol` | 0.495 ± 0.079 | -0.049 | 0.441 0.623 0.512 0.474 0.427 |
| y2 | `a_in_vol` | 0.443 ± 0.044 | -0.101 | 0.401 0.404 0.431 0.494 0.484 |
| y2 | `a_io_vol` | 0.564 ± 0.084 | +0.019 | 0.426 0.650 0.593 0.596 0.554 |
| y2 | `a_io_sd6` | 0.558 ± 0.059 | +0.013 | 0.479 0.606 0.625 0.538 0.539 |
| y2 | `c_n_days_with_tx` | 0.549 ± 0.041 | +0.005 | 0.513 0.539 0.612 0.565 0.517 |
| y3 | `a_vol` | 0.628 ± 0.059 | +0.011 | 0.564 0.715 0.656 0.594 0.614 |
| y3 | `a_in_vol` | 0.633 ± 0.038 | +0.015 | 0.667 0.587 0.637 0.602 0.672 |
| y3 | `a_io_vol` | 0.574 ± 0.024 | -0.044 | 0.580 0.613 0.563 0.566 0.550 |
| y3 | `a_io_sd6` | 0.561 ± 0.073 | -0.057 | 0.672 0.530 0.496 0.597 0.511 |
| y3 | `c_n_days_with_tx` | 0.708 ± 0.038 | +0.090 | 0.646 0.738 0.700 0.715 0.740 |


## 13. vs days (activity twin?)

| col | ρ vs days | n | COPY |
| --- | --- | --- | --- |
| `a_vol` | -0.319 | 15,089 | False |
| `a_in_vol` | -0.274 | 15,089 | False |
| `a_io_vol` | -0.213 | 12,674 | False |
| `a_io_sd6` | -0.164 | 12,674 | False |
| `a_out_vol` | -0.360 | 15,089 | False |
| `a_vol_clip3` | -0.180 | 21,157 | False |


## 14. Y3 size tercile × a_vol quintile

Q5 share in size T1=64.8%. Q5 rate=0.133 body=0.060. T1∩Q5=0.165 notT1∩Q5=0.074. Tail is small-books: **True**.

| size | vol | n | n_pos | P(Y3=1) |
| --- | --- | --- | --- | --- |
| T1 | Q1 | 124 | 10 | 0.081 |
| T1 | Q2 | 147 | 10 | 0.068 |
| T1 | Q3 | 256 | 17 | 0.066 |
| T1 | Q4 | 331 | 39 | 0.118 |
| T1 | Q5 | 546 | 90 | 0.165 |
| T2 | Q1 | 323 | 13 | 0.040 |
| T2 | Q2 | 305 | 12 | 0.039 |
| T2 | Q3 | 316 | 8 | 0.025 |
| T2 | Q4 | 271 | 13 | 0.048 |
| T2 | Q5 | 189 | 12 | 0.063 |
| T3 | Q1 | 396 | 16 | 0.040 |
| T3 | Q2 | 390 | 27 | 0.069 |
| T3 | Q3 | 270 | 19 | 0.070 |
| T3 | Q4 | 240 | 17 | 0.071 |
| T3 | Q5 | 108 | 10 | 0.093 |


## 15. Clip=3 flag

| Y | CV | gap vs size | n_clip | P(Y|clip) | P(Y|body) |
| --- | --- | --- | --- | --- | --- |
| y2 | 0.501 | -0.051 | 1902 | 0.060 | 0.066 |
| y3 | 0.555 | -0.062 | 562 | 0.144 | 0.064 |


## 16. Dark 470 vs invoiced

| slice | n | cos | med a_vol | clip3 |
| --- | --- | --- | --- | --- |
| dark_y2 | 6081 | 459 | 0.839 | 19.3% |
| erp_y2 | 11275 | 736 | 0.831 | 15.2% |
| dark_y3 | 2030 | 262 | 0.617 | 17.5% |
| erp_y3 | 3618 | 463 | 0.611 | 11.1% |

| Y | slice | CV | gap vs size | n |
| --- | --- | --- | --- | --- |
| y2 | dark | 0.437 | -0.138 | 3848 |
| y2 | erp | 0.526 | -0.032 | 7629 |
| y3 | dark | 0.572 | -0.072 | 1499 |
| y3 | erp | 0.673 | +0.048 | 2713 |


## 17. Window / unclip / outflow variants

| Y | feature | CV | gap vs size | vs Javier |
| --- | --- | --- | --- | --- |
| y2 | `a_vol_unclip` | 0.440 | -0.111 | 0.998 SAME |
| y2 | `a_vol_w3` | 0.455 | -0.097 | 0.764 DRIFT |
| y2 | `a_vol_w12` | 0.564 | +0.013 | 0.764 DRIFT |
| y2 | `a_out_vol` | 0.601 | +0.049 | 0.468 DRIFT |
| y2 | `a_sd6_net` | 0.583 | +0.032 | 0.112 DRIFT |
| y3 | `a_vol_unclip` | 0.628 | +0.010 | 0.998 SAME |
| y3 | `a_vol_w3` | 0.625 | +0.007 | 0.764 DRIFT |
| y3 | `a_vol_w12` | 0.631 | +0.013 | 0.764 DRIFT |
| y3 | `a_out_vol` | 0.722 | +0.104 | 0.468 DRIFT |
| y3 | `a_sd6_net` | 0.548 | -0.070 | 0.112 DRIFT |


## 18. Already-neg (B decomp, never X)

Clean-now leftover a_vol CV=0.573 vs size 0.570 (Δ +0.003) n_pos=143. Quintile shape=flat.

| slice | n | n_pos | P(Y2) | med a_vol | clip3 |
| --- | --- | --- | --- | --- | --- |
| already_neg | 1465 | 1039 | 0.709 | 0.840 | 15.4% |
| clean_now | 15891 | 232 | 0.015 | 0.833 | 16.7% |


## 19. Y3 body vs clip=3

| slice | feature | CV | gap vs size | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| body | `a_vol` | 0.579 | +0.044 | 3650 | 232 |
| body | `a_in_vol` | 0.653 | +0.117 | 3650 | 232 |
| body | `c_n_days_with_tx` | 0.711 | +0.175 | 3650 | 232 |
| body | `log1p_a_in3` | 0.536 | +0.000 | 3650 | 232 |
| clip | `a_vol` | 0.500 | -0.235 | 562 | 81 |
| clip | `a_in_vol` | 0.398 | -0.338 | 562 | 81 |
| clip | `c_n_days_with_tx` | 0.731 | -0.004 | 562 | 81 |
| clip | `log1p_a_in3` | 0.735 | +0.000 | 562 | 81 |


## 20. Fold mix (Y3 stressed)

| fold | n | n_pos | P(Y3) | med a_vol | clip3 | med size |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1310 | 54 | 0.041 | 0.721 | 15.0% | 13.017 |
| 1 | 696 | 93 | 0.134 | 0.451 | 12.3% | 13.176 |
| 2 | 1072 | 66 | 0.062 | 0.566 | 9.1% | 13.328 |
| 3 | 1373 | 82 | 0.060 | 0.650 | 16.7% | 13.417 |
| 4 | 1197 | 107 | 0.089 | 0.611 | 11.6% | 13.230 |


## 21. SIZE/COPY for outflow-vol / raw sd / windows

`a_out_vol` vs size ρ=-0.231 SIZE=False COPY=False. `a_sd6_net` vs size ρ=0.739 SIZE=True.

| col | vs | ρ | n | flag |
| --- | --- | --- | --- | --- |
| `a_out_vol` | size | -0.231 | 15,089 | ok |
| `a_out_vol` | a_io_ratio | 0.011 | 15,089 | ok |
| `a_out_vol` | a_growth_3 | -0.101 | 13,532 | ok |
| `a_out_vol` | a_out6 | -0.199 | 15,089 | ok |
| `a_out_vol` | a_vol | 0.468 | 15,089 | ok |
| `a_out_vol` | days | -0.360 | 15,089 | ok |
| `a_sd6_net` | size | 0.739 | 15,089 | SIZE |
| `a_sd6_net` | a_io_ratio | 0.043 | 15,089 | ok |
| `a_sd6_net` | a_growth_3 | 0.030 | 13,532 | ok |
| `a_sd6_net` | a_out6 | 0.835 | 15,089 | COPY |
| `a_sd6_net` | a_vol | 0.112 | 15,089 | ok |
| `a_sd6_net` | days | 0.441 | 15,089 | ok |
| `a_vol_unclip` | size | -0.401 | 15,089 | ok |
| `a_vol_unclip` | a_io_ratio | -0.340 | 15,089 | ok |
| `a_vol_unclip` | a_growth_3 | -0.123 | 13,532 | ok |
| `a_vol_unclip` | a_out6 | -0.163 | 15,089 | ok |
| `a_vol_unclip` | a_vol | 0.998 | 15,089 | COPY |
| `a_vol_unclip` | days | -0.325 | 15,089 | ok |
| `a_vol_w3` | size | -0.316 | 18,729 | ok |
| `a_vol_w3` | a_io_ratio | -0.290 | 18,729 | ok |
| `a_vol_w3` | a_growth_3 | -0.201 | 13,532 | ok |
| `a_vol_w3` | a_out6 | -0.062 | 15,089 | ok |
| `a_vol_w3` | a_vol | 0.764 | 15,089 | ok |
| `a_vol_w3` | days | -0.237 | 18,729 | ok |
| `a_vol_w12` | size | -0.409 | 8,683 | ok |
| `a_vol_w12` | a_io_ratio | -0.326 | 8,683 | ok |
| `a_vol_w12` | a_growth_3 | -0.084 | 7,732 | ok |
| `a_vol_w12` | a_out6 | -0.225 | 8,683 | ok |
| `a_vol_w12` | a_vol | 0.764 | 8,683 | ok |
| `a_vol_w12` | days | -0.377 | 8,683 | ok |
| `a_in_vol` | size | -0.163 | 15,089 | ok |
| `a_in_vol` | a_io_ratio | -0.016 | 15,089 | ok |
| `a_in_vol` | a_growth_3 | -0.164 | 13,532 | ok |
| `a_in_vol` | a_out6 | -0.062 | 15,089 | ok |
| `a_in_vol` | a_vol | 0.473 | 15,089 | ok |
| `a_in_vol` | days | -0.274 | 15,089 | ok |


## 22. a_out_vol honesty

Y3 CV=0.722 Δsize=+0.104 shape=**monotone_up**.

| Y | slice | CV | gap vs size | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| y3 | all | 0.722 | +0.104 | 4212 | 313 |
| y3 | drop12 | 0.720 | +0.102 | 4092 | 313 |
| y3 | dark | 0.763 | +0.119 | 1499 | 105 |
| y3 | erp | 0.720 | +0.095 | 2713 | 208 |
| y3 | body | 0.712 | +0.176 | 3650 | 232 |
| y2 | all | 0.601 | +0.049 | 11477 | 748 |
| y2 | drop12 | 0.586 | +0.042 | 11321 | 614 |
| y2 | dark | 0.614 | +0.039 | 3848 | 332 |
| y2 | erp | 0.549 | -0.009 | 7629 | 416 |
| y2 | body | 0.586 | +0.023 | 9575 | 633 |


Y3 `a_out_vol` quintiles:

| Q | n | n_pos | P(Y3=1) | median |
| --- | --- | --- | --- | --- |
| 1 | 843 | 16 | 0.019 | 0.2237 |
| 2 | 842 | 27 | 0.032 | 0.395 |
| 3 | 842 | 50 | 0.059 | 0.6017 |
| 4 | 842 | 79 | 0.094 | 0.9553 |
| 5 | 843 | 141 | 0.167 | 1.611 |


## 23. a_io_vol Y2 is not the 12 names

Full-panel gap +0.020 → drop12 +0.019 (lost +0.001). Days lost +0.015. not_the_12=**True**. Numeric Y2 KEEP=True — Y3 still loses. Footnote, not a 15-col card.

## 24. a_out_vol Q6 + B leak

| Y | lag0 | lag1 | lift | Q6 KEEP |
| --- | --- | --- | --- | --- |
| y3 | 0.722 | 0.699 | -0.022 | False |
| y2 | 0.601 | 0.615 | +0.015 | False |

| vs | ρ | n | B-copy |
| --- | --- | --- | --- |
| `b_below_0` | -0.066 | 14,968 | False |
| `b_bal_vol` | 0.331 | 14,968 | False |


## 25. a_in_vol body vs clip

Body CV=0.653 Δ=+0.117; clip CV=0.398 Δ=-0.338 (inverted). Overall +0.017 is clip×SIZE mixing. CLOSE.

## 26. Parent return

- `a_vol` (Javier twin): CLOSE. Do not merge.
- `a_io_vol` Y2 footnote (not the 12): True. Y3 loses. Do not merge.
- `a_out_vol` KEEP-shaped? **False**. Different object from Javier vol. Core-stem COPY=False. Parent decides; not a 15-col card tonight.

## 27. vs 15-col shallow-A stems

| vs | ρ all | n | ρ Y3 | COPY |
| --- | --- | --- | --- | --- |
| `c_ss_month` | -0.384 | 15,089 | -0.377 | False |
| `c_salary_month` | -0.334 | 15,089 | -0.312 | False |
| `a_n_tx` | -0.362 | 15,089 | -0.411 | False |
| `f_ds_r` | -0.154 | 15,089 | -0.135 | False |
| `c_n_days_with_tx` | -0.360 | 15,089 | -0.391 | False |


## 28. a_out_vol × size tercile

Q5 share in T1=50.4%. T2+T3 CV=0.726 vs size 0.559 (Δ +0.167) vs days 0.725. KEEP on mid-size: **True** folds 0.727 0.691 0.819 0.697 0.698.

| size | vol | n | n_pos | P(Y3=1) |
| --- | --- | --- | --- | --- |
| T1 | Q1 | 230 | 4 | 0.017 |
| T1 | Q2 | 202 | 15 | 0.074 |
| T1 | Q3 | 215 | 24 | 0.112 |
| T1 | Q4 | 332 | 36 | 0.108 |
| T1 | Q5 | 425 | 87 | 0.205 |
| T2 | Q1 | 296 | 4 | 0.014 |
| T2 | Q2 | 347 | 5 | 0.014 |
| T2 | Q3 | 296 | 15 | 0.051 |
| T2 | Q4 | 249 | 13 | 0.052 |
| T2 | Q5 | 216 | 21 | 0.097 |
| T3 | Q1 | 317 | 8 | 0.025 |
| T3 | Q2 | 293 | 7 | 0.024 |
| T3 | Q3 | 331 | 11 | 0.033 |
| T3 | Q4 | 261 | 30 | 0.115 |
| T3 | Q5 | 202 | 33 | 0.163 |


## 29. acf1 (company median)

| col | n_cos | p10 | p50 | p90 |
| --- | --- | --- | --- | --- |
| `a_vol` | 888 | 0.207 | 0.712 | 0.900 |
| `a_in_vol` | 928 | 0.192 | 0.679 | 0.892 |
| `a_io_vol` | 831 | 0.309 | 0.714 | 0.913 |
| `a_out_vol` | 936 | 0.220 | 0.663 | 0.885 |


## 30. a_out_vol inside size T1

T1 CV=0.676 vs size 0.664 (Δ +0.012) vs days 0.670 n=1404 pos=166 folds 0.606 0.715 0.733 0.600 0.728.

## 31. days × a_out_vol 2×2 (Y3)

Medians days=18 a_out_vol=0.602. Outvol lift in quiet-days +0.098, in busy-days +0.030. Days-substitute: **False**.

| slice | n | n_pos | P(Y3=1) |
| --- | --- | --- | --- |
| low_days_low_out | 723 | 43 | 0.059 |
| low_days_hi_out | 1341 | 211 | 0.157 |
| hi_days_low_out | 1383 | 23 | 0.017 |
| hi_days_hi_out | 765 | 36 | 0.047 |


## 32. days × a_out_vol 2×2 on T2+T3

Quiet-days outvol lift=+0.082 (still not just T1).

| slice | n | n_pos | P(Y3=1) |
| --- | --- | --- | --- |
| low_days_low_out | 431 | 14 | 0.032 |
| low_days_hi_out | 807 | 92 | 0.114 |
| hi_days_low_out | 973 | 15 | 0.015 |
| hi_days_hi_out | 597 | 26 | 0.044 |


## 33. a_out6 × a_out_vol (just high outflow?)

Vol lift inside low `a_out6` = +0.110.

| slice | n | n_pos | P(Y3=1) |
| --- | --- | --- | --- |
| low_out6_low_vol | 1008 | 37 | 0.037 |
| low_out6_hi_vol | 1098 | 161 | 0.147 |
| hi_out6_low_vol | 1098 | 29 | 0.026 |
| hi_out6_hi_vol | 1008 | 86 | 0.085 |


## 34. a_out_vol by so-far

| bucket | n | n_pos | CV | size | gap | low_power |
| --- | --- | --- | --- | --- | --- | --- |
| <6 | 0 | 0 | — | 0.635 | — | True |
| 6-11 | 2287 | 163 | 0.742 | 0.583 | +0.159 | False |
| 12-17 | 1712 | 134 | 0.719 | 0.575 | +0.144 | False |
| 18+ | 213 | 16 | — | — | — | True |


## 35. B diagnostic (never X)

Any |ρ|≥0.80 vs B: **False**.

| pair | ρ | n | B-copy |
| --- | --- | --- | --- |
| b_below_0 | -0.066 | 14,968 | False |
| b_bal_vol | 0.331 | 14,968 | False |
| a_vol~b_below_0 | 0.004 | 14,968 | False |
| a_vol~b_bal_vol | 0.354 | 14,968 | False |


## 36. days × a_out_vol inside dark / invoiced

### dark — quiet-days lift +0.115

| slice | n | n_pos | P(Y3=1) |
| --- | --- | --- | --- |
| low_days_low_out | 228 | 9 | 0.039 |
| low_days_hi_out | 479 | 74 | 0.154 |
| hi_days_low_out | 521 | 12 | 0.023 |
| hi_days_hi_out | 271 | 10 | 0.037 |


### erp — quiet-days lift +0.083

| slice | n | n_pos | P(Y3=1) |
| --- | --- | --- | --- |
| low_days_low_out | 459 | 36 | 0.078 |
| low_days_hi_out | 832 | 134 | 0.161 |
| hi_days_low_out | 897 | 9 | 0.010 |
| hi_days_hi_out | 525 | 29 | 0.055 |


## 37. a_vol (Javier) by so-far

If it never beats size in a window, CLOSE is not a late-trail artifact.

| bucket | n | n_pos | a_vol | size | days | gap |
| --- | --- | --- | --- | --- | --- | --- |
| 6-11 | 2287 | 163 | 0.621 | 0.583 | 0.740 | +0.038 |
| 12-17 | 1712 | 134 | 0.624 | 0.575 | 0.678 | +0.049 |


## 38. who owns the quiet+high-out-vol recoveries?

Labeled pile n=1,341 across 310 companies; Y3=1 n=211 across 96 companies. Top-8 share of recoveries=17.5% (spread).

| company_id | recoveries |
| --- | --- |
| COMP_0279 | 7 |
| COMP_0525 | 6 |
| COMP_0151 | 4 |
| COMP_0066 | 4 |
| COMP_0223 | 4 |
| COMP_0331 | 4 |
| COMP_0198 | 4 |
| COMP_0076 | 4 |


## 39. leftover so-far (why overall a_vol Δ shrinks)

Defined Y3+a_vol n=4,212 CV=0.626 size=0.611 gap=+0.016.

| bucket | n | n_pos | P(Y3=1) | a_vol | size | gap |
| --- | --- | --- | --- | --- | --- | --- |
| <6 | 0 | 0 | — | — | — | — |
| 6-11 | 2287 | 163 | 0.071 | 0.621 | 0.583 | +0.038 |
| 12-17 | 1712 | 134 | 0.078 | 0.624 | 0.575 | +0.049 |
| 18+ | 213 | 16 | 0.075 | — | — | — |
| so_na | 0 | 0 | — | — | — | — |


## 40. a_vol on T2+T3 (Javier off small books)

n=2,808 pos=147 a_vol=0.532 size=0.559 days=0.725 gap=-0.027 window-KEEP=False. Overall still CLOSE (Δ=+0.009).

## 41. a_vol inside size T1

n=1,404 pos=166 a_vol=0.636 size=0.664 days=0.670 gap=-0.028 T1-KEEP=False. Both slices lose to size — overall +0.009 is between-tercile.

## 42. Simpson: size tercile × a_vol median

a_vol median=0.612. Within each tercile, hi-vol vs low-vol is the test; tercile base rates show the between-group mix.

| tercile | n | n_pos | P(Y3=1) | mean a_vol |
| --- | --- | --- | --- | --- |
| T1 | 1404 | 166 | 0.118 | 1.554 |
| T2 | 1404 | 58 | 0.041 | 0.851 |
| T3 | 1404 | 89 | 0.063 | 0.659 |

| slice | n | n_pos | P(Y3=1) | mean a_vol |
| --- | --- | --- | --- | --- |
| T1_low_vol | 382 | 26 | 0.068 | 0.334 |
| T1_hi_vol | 1022 | 140 | 0.137 | 2.010 |
| T2_low_vol | 801 | 30 | 0.037 | 0.307 |
| T2_hi_vol | 603 | 28 | 0.046 | 1.574 |
| T3_low_vol | 923 | 51 | 0.055 | 0.287 |
| T3_hi_vol | 481 | 38 | 0.079 | 1.372 |


## 43. a_out_vol Simpson (within-tercile lift)

a_out_vol median=0.602.

| slice | n | n_pos | P(Y3=1) | hi-lo |
| --- | --- | --- | --- | --- |
| T1_low | 531 | 30 | 0.056 |  |
| T1_hi | 873 | 136 | 0.156 | +0.099 |
| T2_low | 800 | 16 | 0.020 |  |
| T2_hi | 604 | 42 | 0.070 | +0.050 |
| T3_low | 775 | 20 | 0.026 |  |
| T3_hi | 629 | 69 | 0.110 | +0.084 |


## 44. a_vol vs clip=3 flag

If the flag matches continuous AUROC, Javier vol is mostly the cap.

| y | flag | continuous | size | almost_flag | n | n_pos |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 0.555 | 0.626 | 0.617 | False | 5648 | 402 |
| y2_neg_2of3 | 0.501 | 0.439 | 0.551 | False | 17356 | 1271 |


## 45. company-mean vol vs ever Y3=1

725 train companies with a labeled Y3 month; 174 ever recover. Group-fold on company rows (fold of first labeled month).

| feature | company AUROC | n | n_pos |
| --- | --- | --- | --- |
| a_vol | 0.591 | 600 | 143 |
| a_out_vol | 0.683 | 600 | 143 |
| size | 0.631 | 708 | 169 |
| days | 0.666 | 725 | 174 |


## 46. company-mean on common companies

600 companies with all four means defined; 143 ever recover. a_out_vol vs size +0.071; vs days +0.002. a_vol vs size -0.022.

| feature | company AUROC | n | n_pos |
| --- | --- | --- | --- |
| a_vol | 0.591 | 600 | 143 |
| a_out_vol | 0.683 | 600 | 143 |
| size | 0.613 | 600 | 143 |
| days | 0.681 | 600 | 143 |


## 47. company-demeaned (timing vs trait)

If demeaning kills AUROC, the signal is a sticky company type. If it survives, there is within-company timing.

| feature | raw | demean | drop | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| a_vol | 0.626 | 0.465 | +0.162 | 4212 | 313 |
| a_out_vol | 0.722 | 0.549 | +0.172 | 4212 | 313 |
| c_n_days_with_tx | 0.711 | 0.502 | +0.209 | 5648 | 402 |
| log1p_a_in3 | 0.617 | 0.624 | -0.007 | 5528 | 391 |


## 48. company-mean copy screen

`a_out_vol` vs days ρ=-0.453 COPY=False. If COPY, the company-trait KEEP is a days twin at company grain.

| a | b | ρ | n | COPY |
| --- | --- | --- | --- | --- |
| a_out_vol | c_n_days_with_tx | -0.453 | 600 | False |
| a_out_vol | log1p_a_in3 | -0.242 | 600 | False |
| a_out_vol | a_vol | 0.487 | 600 | False |
| a_vol | c_n_days_with_tx | -0.326 | 600 | False |
| a_vol | log1p_a_in3 | -0.460 | 600 | False |


## 49. company-mean days × a_out_vol vs ever-recover

600 companies. Quiet-days lift +0.132; busy-days lift +0.131. Complementary if both >0.

| slice | n | ever | P(ever Y3) |
| --- | --- | --- | --- |
| low_days_low_out | 105 | 26 | 0.248 |
| low_days_hi_out | 195 | 74 | 0.379 |
| hi_days_low_out | 195 | 19 | 0.097 |
| hi_days_hi_out | 105 | 24 | 0.229 |


## 50. company-mean vs ever Y2

900 companies; 130 ever Y2. Footnote only.

| feature | company AUROC | n | n_pos |
| --- | --- | --- | --- |
| a_vol | 0.476 | 900 | 130 |
| a_out_vol | 0.550 | 900 | 130 |
| a_io_vol | 0.521 | 900 | 130 |
| size | 0.550 | 900 | 130 |


## 51. store audit

Named vol columns already in `monthly.parquet`: none. Clean (do not merge) = True.

## 52. a_in_vol company-mean Y3

600 companies; 143 ever recover. a_in_vol vs size +0.001 (month-level missed KEEP by 0.003).

| feature | company AUROC | n | n_pos |
| --- | --- | --- | --- |
| a_in_vol | 0.614 | 600 | 143 |
| a_vol | 0.591 | 600 | 143 |
| size | 0.613 | 600 | 143 |


## 53. activity twin screen

Any COPY vs a_n_tx / days? **False**.

| col | vs | ρ | n | COPY |
| --- | --- | --- | --- | --- |
| a_out_vol | a_n_tx | -0.362 | 15,089 | False |
| a_out_vol | c_n_days_with_tx | -0.360 | 15,089 | False |
| a_vol | a_n_tx | -0.340 | 15,089 | False |
| a_vol | c_n_days_with_tx | -0.319 | 15,089 | False |
| a_in_vol | a_n_tx | -0.284 | 15,089 | False |


## 54. company-mean a_out_vol vs outflow level

COPY vs a_out6 / a_op_out? **False**.

| vs | ρ | n | COPY |
| --- | --- | --- | --- |
| a_out6 | -0.124 | 600 | False |
| a_op_out | -0.081 | 600 | False |


## 55. Q5 recovery overlap (same tail?)

Y3 pos=313; a_vol Q5 pos=112; a_out_vol Q5 pos=141; both=72; out-only=69; vol-only=40. both/outQ5=51.1% same_tail=False.

## 56. size mix of out-Q5-only recoveries

If out-only recoveries sit in T2+T3, a_out_vol is not recycling Javier's small-book tail.

| slice | n | T1 | T2 | T3 | T2+T3 |
| --- | --- | --- | --- | --- | --- |
| out_only | 69 | 29 | 16 | 24 | 58.0% |
| both_q5 | 72 | 58 | 5 | 9 | 19.4% |
| all_pos | 313 | 166 | 58 | 89 | 47.0% |


## 57. a_out_vol after dropping a_vol Q5

n=3,369 pos=201 a_out_vol=0.720 size=0.453 days=0.721 gap=+0.266 KEEP-shaped=True.

## 58. a_vol after dropping a_out_vol Q5

n=3,369 pos=172 a_vol=0.635 size=0.581 days=0.716 gap=+0.054 KEEP-shaped=True. Slice-only; overall a_vol stays CLOSE. Days still 0.716.

## 59. holdout coverage (LOW_POWER)

72 companies (72=True), CM=1,073. a_vol 66.5%; a_out_vol 66.5%; Y3 pos=14/235. No AUROC claim.

## 60. leak + parquet untouched

leakage_check a_vol* + a_out_vol vs Y3 forbidden B: ok=True. `monthly.parquet` age 4.2h (not rewritten this run).

## 61. registry rows this agent

21 append-only rows for `6b456387` / `a_vol_qa`.

## 62. score_pipeline quote (read-only)

Line 165: `P["volatility"] = np.minimum(3, P["sd6"] / np.maximum(P["in6"], 1))` still present=True. File not edited.

## 63. a_out_vol Y3 fold-min

Folds `0.710 0.724 0.782 0.664 0.728`; min=0.664; invert(<0.50)=False.

## 64. a_vol fold-min

| Y | CV | folds | min | invert |
| --- | --- | --- | --- | --- |
| y3 | 0.626 | 0.552 0.715 0.656 0.594 0.614 | 0.552 | False |
| y2 | 0.439 | 0.429 0.377 0.488 0.474 0.427 | 0.377 | True |


## 65. a_out_vol Y2 fold-min

CV=0.601 Δsize=+0.049 folds `0.603 0.648 0.668 0.443 0.642` min=0.443 invert=True. Y3-only KEEP-shape.

## 66. confirm or kill a_out_vol 0.722

**Reproduced 0.722?** **True** (CV=0.722 ± 0.042 n=4,212 pos=313 folds `0.710 0.724 0.782 0.664 0.728`).

**X = CLOSE**. **Y = PARK**. Merge **NO**. company-style dummy (demean kills; high ICC); like uncat ICC — PARK as health Y. Not a 15-col card. Night Y3 GBM quote 0.762 / 0.752 untouched.

- SIZE: vs `log1p(a_in3)` / `a_op_in` fail=False. Y3 Δ vs size +0.104 (size=0.617 days=0.711).
- Leak twin any |ρ|≥0.80 on Y3-labeled: **False**.
- Company ρ vs days on Y3-labeled companies: -0.451 n=600 (quote −0.453).
- Drop 12 chronic: CV=0.720 Δ=+0.002 move≥0.02=False (n_ids=12).
- Short <12 so-far: 0.742 vs size 0.600 n=2287 pos=163.
- Long ≥12: 0.724 vs size 0.589 n=1925 pos=150.
- Q6 lag1=0.699 vs now=0.722 lift=-0.022 KEEP=False.
- Month 2×2 quiet lift +0.098; hot cell is 12 names=False.
- Company 2×2 quiet lift +0.104 (quote +13pp); hot cell is 12 names=False.
- Demean 0.722→0.549 drop=+0.172. ICC=0.751 η²=0.741 on 600 companies. **trait=True shock=False**.
- later-store KEEP=False (needs month shock; this is a company trait).
- Drop company-Q5: Δ=+0.066 move=True; Q5-only CV=0.540; Q5 dummy=0.647 vs continuous 0.722.
- Pass49-style company 2×2 +13.2pp (n=195/105). Neither cell is the 12 names.

### leak on Y3-labeled months

| vs | ρ | n | twin≥0.80 | SIZE≥0.50 |
| --- | --- | --- | --- | --- |
| c_n_days_with_tx | -0.391 | 4,212 | False | False |
| c_ss_month | -0.377 | 4,212 | False | False |
| c_salary_month | -0.312 | 4,212 | False | False |
| a_n_tx | -0.411 | 4,212 | False | False |
| a_out6 | -0.100 | 4,212 | False | False |
| a_io_ratio | -0.121 | 4,212 | False | False |
| log1p_a_in3 | -0.173 | 4,212 | False | False |
| a_op_in | -0.276 | 4,212 | False | False |


### month 2×2 (days × a_out_vol medians)

| slice | n | n_pos | P(Y3=1) | chronic n | chronic pos | ch share |
| --- | --- | --- | --- | --- | --- | --- |
| low_days_low_out | 723 | 43 | 0.059 | 19 | 0 | 2.6% |
| low_days_hi_out | 1341 | 211 | 0.157 | 4 | 0 | 0.3% |
| hi_days_low_out | 1383 | 23 | 0.017 | 63 | 0 | 4.6% |
| hi_days_hi_out | 765 | 36 | 0.047 | 34 | 0 | 4.4% |


### company 2×2 (mean days × mean a_out_vol)

| slice | n | ever | P(ever) | chronic cos | chronic ever |
| --- | --- | --- | --- | --- | --- |
| low_days_low_out | 107 | 26 | 0.243 | 1 | 0 |
| low_days_hi_out | 193 | 67 | 0.347 | 0 | 0 |
| hi_days_low_out | 193 | 17 | 0.088 | 5 | 0 |
| hi_days_hi_out | 107 | 24 | 0.224 | 6 | 0 |


## 67. +13pp reconcile + group ICC

Pass49-style company 2×2 quiet lift +0.132 (37.9% vs 24.8%, n=195/105) quote +13pp **True**.

ICC company=0.751 η²=0.741; ICC group=0.486 η²=0.282 on 134 groups. High-out-vol companies=300 in 104 groups.

## 68. first defined month vs company mean

first↔mean ρ=0.859 n=600. Company AUROC first=0.625 mean=0.682 pos=134. If first ≈ mean, the type is set at month 6.

## 69. within-company residual shock

422 companies with both sides. Above own median P=0.045 n=2044; below P=0.058 n=1712; lift=-0.012. Residual shock=False.

## 70. hands off

Night Y3 GBM quote 0.762 / 0.752 untouched=True. companies_qa / factoring_qa not edited.

## 71. company-mean a_out_vol quintiles vs ever Y3

monotone_up=False.

| Q | n | ever | P(ever) | median |
| --- | --- | --- | --- | --- |
| 1 | 120 | 12 | 0.100 | 0.301 |
| 2 | 120 | 17 | 0.142 | 0.492 |
| 3 | 120 | 29 | 0.242 | 0.745 |
| 4 | 120 | 23 | 0.192 | 1.065 |
| 5 | 120 | 53 | 0.442 | 1.617 |


## 72. company Q5 pile — dark vs invoiced

| slice | n | ever | P(ever) |
| --- | --- | --- | --- |
| q5_dark | 53 | 22 | 0.415 |
| q5_erp | 67 | 31 | 0.463 |
| not_q5 | 480 | 81 | 0.169 |


## 73. ICC all-train vs Y3-labeled

All-train ICC=0.624 η²=0.583 n=15,089 cos=1211. Y3-labeled ICC=0.751 η²=0.741 n=4,212 cos=600. High either way.

## 74. chronic 12 on Y3

162 labeled months, 0 recoveries, 192 with a_out_vol. zero_pos=True. Ids: COMP_0188, COMP_0254, COMP_0356, COMP_0577, COMP_0679, COMP_0750, COMP_0885, COMP_0910, COMP_0911, COMP_0947, COMP_0969, COMP_1213.

## 75. a_out_vol coverage by so-far

| bucket | CM | a_out_vol | Y3 labeled |
| --- | --- | --- | --- |
| <6 | 6068 | 0.0% | 1436 |
| 6-11 | 6406 | 100.0% | 2287 |
| 12+ | 8683 | 100.0% | 1925 |


## 76. Y3 labeled without a_out_vol

Defined n=4,212 pos=313 P=0.074. Missing n=1,436 pos=89 P=0.062. 0.722 is only on months with 6+ history.

## 77. ICC on companies with ≥8 labeled months

282 companies, n=3,085, ICC=0.686 η²=0.716, CV=0.690 pos=91. Trait holds on long books.

## 78. owned files

This process only writes a_vol_qa.py / a_vol_qa.md / wave4_a_vol.md / registry / PNG.

## 79. leak on all train months

Any twin |ρ|≥0.80 off the Y3 path? **False**.

| vs | ρ | n | twin |
| --- | --- | --- | --- |
| c_n_days_with_tx | -0.360 | 15,089 | False |
| c_ss_month | -0.384 | 15,089 | False |
| c_salary_month | -0.334 | 15,089 | False |
| a_n_tx | -0.362 | 15,089 | False |
| a_out6 | -0.199 | 15,089 | False |
| a_io_ratio | 0.011 | 15,089 | False |
| log1p_a_in3 | -0.231 | 15,089 | False |
| a_op_in | -0.278 | 15,089 | False |


## 80. CV vs in-sample

OOF CV=0.722 in-sample train=0.722 (5 folds). Quote 0.722 is the OOF mean.

## 81. leave-one-fold of 0.722

Min leave-one-fold mean=0.706 still≥0.70=**True**. 0.722 is not one lucky fold.

## 82. fold signs

signs=[1, 1, 1, 1, 1] all_same=**True** train_sign=1.

## 83. company-grain OOF (trait transfer)

Company-mean a_out_vol vs ever-Y3 group-fold CV=0.682 n=600 pos=134 folds `0.654 0.702 0.723 0.641 0.690`. Transfers as a company type across groups=**True**. Still not a month shock — new companies need their own history.

## 84. night Y3 quote files

0.762 / 0.752 still present in quote files=**True**. This process did not write them.

## 85. company-OOF vs days / size

out=0.683 days=0.666 size=0.631 (Δdays=+0.017 Δsize=+0.052). Trait is complementary to days at company grain, not a days dummy.

## 86. Pearson leak (Y3-labeled)

any twin=False any SIZE=False.

| vs | Pearson ρ | n | twin | SIZE |
| --- | --- | --- | --- | --- |
| c_n_days_with_tx | -0.409 | 4,212 | False | False |
| c_ss_month | -0.405 | 4,212 | False | False |
| c_salary_month | -0.346 | 4,212 | False | False |
| a_n_tx | -0.223 | 4,212 | False | False |
| a_out6 | -0.020 | 4,212 | False | False |
| a_io_ratio | 0.022 | 4,212 | False | False |
| log1p_a_in3 | -0.238 | 4,212 | False | False |
| a_op_in | 0.004 | 4,212 | False | False |


## 87. parquet + sibling mtimes

This process did not write parquet / companies_qa / factoring_qa.

| file | age_min |
| --- | --- |
| data/feature_store/monthly.parquet | 251.1 |
| data/feature_store/targets.parquet | 236.4 |
| analysis/evaluate/companies_qa.py | 13.6 |
| analysis/evaluate/factoring_qa.py | 19.4 |


## 88. 12 chronic names

n=12: `COMP_0188, COMP_0254, COMP_0356, COMP_0577, COMP_0679, COMP_0750, COMP_0885, COMP_0910, COMP_0911, COMP_0947, COMP_0969, COMP_1213`. Y3 labeled=162 pos=0.

## 89. drop company-Q5 pile

Drop 120 Q5 companies: CV=0.655 vs full 0.722 Δ=+0.066 move≥0.02=**True** n=3567 pos=178. If this moves, the 0.722 is the high-vol company pile (trait).

## 90. product/ hands off

product/ exists=True age_h=6.8. This process did not write it. No 0–100.

## 91. holdout 72 coverage

a_out_vol=66.5% Y3 pos=14. LOW_POWER — coverage only, no AUROC.

## 92. max |Spearman| leak

max |ρ| vs `a_n_tx` = -0.411 twin=False.

## 93. leftover after Q5 vs size

CV=0.655 size=0.585 Δ=+0.070 still≥0.02=**True** n=3567 pos=178.

## 94. Q5 group spread

Q5 n=120 groups=71 top share=5.0% one-group dummy=**False**.

## 95. Q5 ∩ 12 chronic

overlap=0 of 120. ids=`none`.

## 96. no LightGBM in this module

ok=**True** hits=none.

## 97. company-OOF Q1–Q4 only

CV=0.604 n=480 pos=81 folds `0.562 0.632 0.626 0.591 0.610`. Q5-vs-rest step=**False**.

## 98. drop company Q4+Q5

n_drop=240 CV=0.675 size=0.605 Δ=+0.070 n=2789 pos=128.

## 99. body ICC + demean (Q1–Q4)

body CV=0.655 demean=0.648 drop=+0.007 ICC=0.567 η²=0.590 n_cos=480 trait=**False**.

## 100. Q5-only month AUROC

Q5-only CV=0.540 demean=0.523 n=645 pos=135 folds `0.487 0.511 0.569 0.584 0.548`. Ranks inside the pile=**False**. The quoted 0.722 is Q5-vs-rest (company type), not a body month shock.

## 101. Q5 dummy vs continuous

Q5 dummy CV=0.647 continuous=0.722 Δ=+0.075 dummy-shaped=**False**.

## 102. Q5 share by fold

| fold | n | n_pos | Q5 share | Q5 of pos |
| --- | --- | --- | --- | --- |
| 0 | 995 | 41 | 10.2% | 29.3% |
| 1 | 504 | 72 | 19.6% | 50.0% |
| 2 | 787 | 48 | 10.8% | 52.1% |
| 3 | 1081 | 71 | 19.0% | 33.8% |
| 4 | 845 | 81 | 18.3% | 46.9% |


## 103. owned files

This process writes `a_vol_qa.py` / `a_vol_qa.md` / `wave4_a_vol.md` only. git porcelain captured (ok=True).

## 104. parent return line

`PARENT reproduced=True CV=0.722 leak_twin=False SIZE=False max_leak=-0.411 vs a_n_tx co_days=-0.451 drop12=+0.002 short=0.742 long=0.724 Q6=False plus13=True trait=True shock=False Q5drop=+0.066 loo=0.706 X=CLOSE Y=PARK merge=NO`

## 105. night quote file ages

All quote files older than this stay=**True**.

## 106. Spearman vs Pearson max leak

Spearman max `a_n_tx` -0.411; Pearson max `c_n_days_with_tx` -0.409; any twin=False.

## 107. permutation of company means

obs AUROC=0.688 n=600 pos=134 p=0.0010 (1000 perms).

## 108. Javier a_vol still CLOSE

Y3 CV=0.626 Δsize=+0.009 still_CLOSE=**True**. Do not merge.

## 109. PNG + later-store KEEP

PNG exists=True bytes=124708 later_keep=**False**.

## 110. wave self-check

ok=**True** hits={'reproduced': True, 'close': True, 'cv722': True, 'quote762': True, 'nomerge': True}.

## 111. company-mean AUROC bootstrap

p2.5/p50/p97.5=0.634/0.686/0.741 n_boot=200.

## 112. resume table self-check

ok=**True** hits={'reproduced': True, 'close': True, 'trait': True, 'leak': True}.

## 113. +13pp cells not n=12

n=195/105 plus13=True tiny12=**False**.

## 114. exact night quotes

0.7622 in wave3=True; 0.7520 in i_lift=True. ok=**True**.

## 115. no sibling / parquet / product writes

ok=**True** bad=none.

## 116. parquet age

monthly.parquet age_h=4.19 ok=**True**.

## 117. holdout vs train Q5 cutoff (coverage only)

train Q5 cut=1.353; holdout cos defined=71/72; at/above Q5=21 (29.6%). LOW_POWER — no AUROC.

## 118. wave holdout mention

wave mentions holdout=**True** (3027 chars).

## 119. stay clock

minutes since 03:47=30.4 ≥30=**True** (2026-09-19T04:17:23.769687+02:00).

## 120. leak one-liner

`c_n_days_with_tx=-0.391 c_ss_month=-0.377 c_salary_month=-0.312 a_n_tx=-0.411 a_out6=-0.100 a_io_ratio=-0.121 log1p_a_in3=-0.173 a_op_in=-0.276` twin=False SIZE=False.

## What this is not

- Not a 0–100. Not a store merge. Not a 15-col retrain.
- Not family B as Y2/Y3 X.
- Uncat / FX modules were not edited.

Elapsed 15.3s. PNG: `a_vol_vs_b_bal_vol.png` (True).
