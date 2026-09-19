# Y2 why — same crash as Y4, or a different turn?

- **When:** 2026-09-19T02:50
- **Agent:** `1bb2643e`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.evaluate.y2_why`
- **Holdout:** 72 companies, seed 20260918. Coverage only. Rates + singles on train.
- **Y:** `y2_neg_2of3` only from `targets.parquet` (train 17356 / 1271 / **7.32%**). Rejected Y2 columns not revived.
- **X candidates:** `a_io_ratio` / `a_out6` / `log1p(a_in3)` / `c_n_days_with_tx` / `d_cust_hhi` / `f_ds_r` / `c_zero_in_month`. **Never B.** `b_liq` / `b_runway` / `b_below_0` are decomp + leak only.
- **Quote:** train group-fold CV (5 folds). Night GBM PARK 0.540±0.117 (fold 0 = 0.344) is the bar to *explain*, not beat with a tree.
- **Brief:** Q3 turning (82→68) / Q5 why. Q6 only via honest 1-month lag.
- Not bankruptcy. Not a 0–100.

## Decision

**CLOSE.** Y2 is a different turn from Y4 crash (Y2pos crash=0.354, Y2∩Y4=0.127, Y4∩Y2=0.095). c_n_days_with_tx CV 0.571 gap_vs_size=+0.031 shape=flat — no KEEP. Trees stay PARK.

Trees stay **PARK**.

## Pass 1 — decompose positives

High `a_out6` / `d_cust_hhi` = that company's own expanding p80 (months ≤ t, min 6). Inflow crash uses future `a_in3` to *name* the label — not X. `y3_stressed_now` reads family B to decompose the Y (the Y *is* B) — never as X.

| flag | n_hi / n_defined pos | share of pos | coverage of pos |
|---|---:|---:|---:|
| `y4_ds_r_double` | 31 / 245 | 0.127 | 0.193 |
| `y9_fee_r_ownp80` | 92 / 604 | 0.152 | 0.475 |
| `y3_stressed_now` | 1180 / 1271 | 0.928 | 1.000 |
| `y3_labeled` | 914 / 1271 | 0.719 | 1.000 |
| `a_out6_hi` | 166 / 437 | 0.380 | 0.344 |
| `inflow_crash` | 344 / 971 | 0.354 | 0.764 |
| `d_cust_hhi_hi` | 50 / 231 | 0.216 | 0.182 |
| `hhi_tail` | 67 / 445 | 0.151 | 0.350 |
| `c_zero_in_month` | 95 / 1271 | 0.075 | 1.000 |
| `b_below_0` | 1039 / 1271 | 0.817 | 1.000 |

Positives median in-ratio **1.03** (neg 0.99). Y4 crash positives sat at 0.36 — compare.

### 2×2 overlap

| pair | n | both | a only | b only | neither | share a∈b | share b∈a | Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| y2_x_crash (`y2_neg_2of3` × `inflow_crash`) | 13471 | 344 | 627 | 4728 | 7772 | 0.354 | 0.068 | 0.060 |
| y2_x_y4 (`y2_neg_2of3` × `y4_ds_r_double`) | 2368 | 31 | 214 | 297 | 1826 | 0.127 | 0.095 | 0.057 |
| y2_x_y9 (`y2_neg_2of3` × `y9_fee_r_ownp80`) | 9551 | 92 | 512 | 1257 | 7690 | 0.152 | 0.068 | 0.049 |
| crash_x_y4 (`inflow_crash` × `y4_ds_r_double`) | 2299 | 261 | 453 | 63 | 1522 | 0.366 | 0.806 | 0.336 |

Same-crash-as-Y4 flag: **False** (needs ≥70% of Y2 pos crashing *and* ≥50% of those also Y4).

## Pass 2 — quintiles (train labeled cuts)

### `a_io_ratio`

n=14968 bins=5 shape=**flat** monotone↑=False tail_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-1696.131, 0.238] | 2994 | 213 | 0.071 | 0 |
| 2 | (0.238, 0.82] | 2993 | 225 | 0.075 | 0.6015 |
| 3 | (0.82, 1.079] | 2994 | 177 | 0.059 | 0.9747 |
| 4 | (1.079, 1.919] | 2993 | 172 | 0.057 | 1.306 |
| 5 | (1.919, 3.0] | 2994 | 257 | 0.086 | 3 |

### `a_out6`

n=11477 bins=5 shape=**flat** monotone↑=False tail_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-28157774.900999997, 44367.63] | 2296 | 159 | 0.069 | 3229 |
| 2 | (44367.63, 354157.184] | 2295 | 92 | 0.040 | 1.46e+05 |
| 3 | (354157.184, 1211746.286] | 2295 | 126 | 0.055 | 6.742e+05 |
| 4 | (1211746.286, 4292621.67] | 2295 | 180 | 0.078 | 2.232e+06 |
| 5 | (4292621.67, 17114407320.57] | 2296 | 191 | 0.083 | 1.234e+07 |

### `log1p_a_in3`

n=14968 bins=5 shape=**flat** monotone↑=False tail_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 9.689] | 2994 | 160 | 0.053 | 0.004975 |
| 2 | (9.689, 11.859] | 2993 | 185 | 0.062 | 10.98 |
| 3 | (11.859, 13.133] | 2994 | 225 | 0.075 | 12.51 |
| 4 | (13.133, 14.449] | 2993 | 244 | 0.082 | 13.77 |
| 5 | (14.449, 22.713] | 2994 | 230 | 0.077 | 15.3 |

### `c_n_days_with_tx`

n=17356 bins=5 shape=**flat** monotone↑=False tail_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 5.0] | 4009 | 225 | 0.056 | 2 |
| 2 | (5.0, 11.0] | 3270 | 170 | 0.052 | 9 |
| 3 | (11.0, 17.0] | 3185 | 185 | 0.058 | 15 |
| 4 | (17.0, 22.0] | 3718 | 402 | 0.108 | 20 |
| 5 | (22.0, 31.0] | 3174 | 289 | 0.091 | 27 |

### `d_cust_hhi`

n=7130 bins=5 shape=**flat** monotone↑=False tail_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (0.0017599999999999998, 0.148] | 1426 | 78 | 0.055 | 0.07025 |
| 2 | (0.148, 0.359] | 1426 | 110 | 0.077 | 0.2484 |
| 3 | (0.359, 0.628] | 1426 | 131 | 0.092 | 0.4847 |
| 4 | (0.628, 0.979] | 1426 | 63 | 0.044 | 0.8288 |
| 5 | (0.979, 1.0] | 1426 | 63 | 0.044 | 1 |

### `f_ds_r`

n=14968 bins=2 shape=**monotone_up** monotone↑=True tail_only=None

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 0.0203] | 11974 | 755 | 0.063 | 0 |
| 2 | (0.0203, 2.0] | 2994 | 289 | 0.097 | 0.1453 |

Plot: `y2_why_quintiles.png`.

## Pass 3 — single-feature CV

Size `log1p(|a_in3|)` train AUROC **0.540** (two-sided 0.540). Size ≥ 0.60 → PARK as X. Night GBM 0.540 is the bar to explain.

| feature | CV AUROC ± sd | train | sign | coverage | gap vs size | gap vs 0.540 | folds |
|---|---:|---:|---:|---:|---:|---:|---|
| `a_io_ratio` | 0.473 ± 0.069 | 0.509 | +1 | 0.862 | -0.067 | -0.067 | 0.438 0.571 0.481 0.384 0.490 |
| `a_out6` | 0.561 ± 0.092 | 0.550 | +1 | 0.661 | +0.021 | +0.021 | 0.613 0.680 0.566 0.447 0.501 |
| `log1p_a_in3` | 0.551 ± 0.046 | 0.540 | +1 | 0.862 | +0.012 | +0.011 | 0.523 0.609 0.594 0.519 0.511 |
| `c_n_days_with_tx` | 0.571 ± 0.046 | 0.577 | +1 | 1.000 | +0.031 | +0.031 | 0.623 0.539 0.612 0.565 0.517 |
| `d_cust_hhi` | 0.564 ± 0.161 | 0.535 | -1 | 0.411 | +0.024 | +0.024 | 0.707 0.699 0.325 0.600 0.488 |
| `f_ds_r` | 0.538 ± 0.066 | 0.539 | +1 | 0.862 | -0.002 | -0.002 | 0.539 0.621 0.438 0.532 0.557 |
| `c_zero_in_month` | 0.522 ± 0.027 | 0.521 | -1 | 1.000 | -0.018 | -0.018 | 0.526 0.539 0.555 0.504 0.486 |

Best legal single: `c_n_days_with_tx` CV **0.571**.

## Pass 4 — why fold 0 inverted

fold0 rate 0.111 vs rest 0.063; late 0.615 vs 0.640; short 0.270 vs 0.283; dark 0.407 vs 0.379; monopoly 0.270 vs 0.227; med group_size 11.0 vs 10.8; med months_on_book 24.0 vs 23.0

| fold | n | n_pos | P(Y=1) | cos | groups | late cos | short <12 cos | dark cos | HHI>0.975 cos | med group_size | med months-on-book | mean log1p(in3) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3384 | 374 | 0.111 | 226 | 47 | 0.615 | 0.270 | 0.407 | 0.270 | 11.0 | 24.0 | 11.52 |
| 1 | 3158 | 89 | 0.028 | 243 | 47 | 0.646 | 0.379 | 0.576 | 0.165 | 8.0 | 24.0 | 10.80 |
| 2 | 3340 | 187 | 0.056 | 232 | 47 | 0.655 | 0.276 | 0.336 | 0.203 | 11.0 | 24.0 | 11.23 |
| 3 | 3787 | 310 | 0.082 | 236 | 47 | 0.504 | 0.208 | 0.284 | 0.220 | 11.0 | 24.0 | 11.79 |
| 4 | 3687 | 311 | 0.084 | 258 | 47 | 0.756 | 0.271 | 0.318 | 0.322 | 13.0 | 20.0 | 11.41 |

Night tree fold 0 = 0.344. This table is the group mix, not a new fit.

## Pass 5 — leak vs family B

Fail if Spearman |ρ| ≥ 0.8 vs `b_liq` / `b_runway` / `b_below_0` (that is the Y).

| feature | vs | n | Spearman | Pearson | fail |
|---|---|---:|---:|---:|---|
| `a_io_ratio` | `b_liq` | 14968 | +0.091 | +0.001 | False |
| `a_io_ratio` | `b_runway` | 14968 | +0.124 | -0.000 | False |
| `a_io_ratio` | `b_below_0` | 14968 | +0.014 | +0.005 | False |
| `a_out6` | `b_liq` | 11477 | +0.388 | +0.203 | False |
| `a_out6` | `b_runway` | 11477 | -0.432 | -0.031 | False |
| `a_out6` | `b_below_0` | 11477 | +0.029 | -0.011 | False |
| `log1p_a_in3` | `b_liq` | 14968 | +0.406 | +0.114 | False |
| `log1p_a_in3` | `b_runway` | 14968 | -0.328 | -0.421 | False |
| `log1p_a_in3` | `b_below_0` | 14968 | +0.036 | +0.034 | False |
| `c_n_days_with_tx` | `b_liq` | 17356 | +0.182 | +0.007 | False |
| `c_n_days_with_tx` | `b_runway` | 14968 | -0.314 | -0.349 | False |
| `c_n_days_with_tx` | `b_below_0` | 17356 | +0.074 | +0.071 | False |
| `d_cust_hhi` | `b_liq` | 7130 | -0.070 | -0.082 | False |
| `d_cust_hhi` | `b_runway` | 6746 | +0.057 | +0.116 | False |
| `d_cust_hhi` | `b_below_0` | 7130 | -0.026 | -0.033 | False |
| `f_ds_r` | `b_liq` | 14968 | +0.009 | -0.022 | False |
| `f_ds_r` | `b_runway` | 14968 | -0.235 | -0.023 | False |
| `f_ds_r` | `b_below_0` | 14968 | +0.034 | +0.033 | False |
| `c_zero_in_month` | `b_liq` | 17356 | -0.138 | -0.022 | False |
| `c_zero_in_month` | `b_runway` | 14968 | +0.244 | +0.320 | False |
| `c_zero_in_month` | `b_below_0` | 17356 | -0.041 | -0.041 | False |

Illegal B-copies: **0**.

## Pass 6 — dark 470 vs invoiced 744

Join-QA population (DuckDB book invoices). Confirm only. Not a new Y.

| slice | n | n_pos | companies | base rate | size AUROC |
|---|---:|---:|---:|---:|---:|
| train_all | 17356 | 1271 | 1195 | 0.073 | 0.540 |
| dark_470 | 6081 | 556 | 459 | 0.091 | 0.543 |
| invoiced_744 | 11275 | 715 | 736 | 0.063 | 0.530 |

CONFIRM 9.14% vs 6.34%: **True** (dark cos=470, invoiced=744, confirm_470=True).

## Mapping (Q3 / Q5 / Q6)

Y2 `neg_2of3` is the brief's 82→68 *direction* (Q3), but on train it is mostly **already-negative persistence** (82% of positives are `b_below_0` at t; already-neg months run at 71% Y2 vs clean-now **1.5%**). It is **not** Y4's inflow crash (Y2pos crash=35%, Y2∩Y4=13%, Y4∩Y2=9%; median in-ratio 1.03 vs Y4's 0.36). No non-B Q5 KEEP: best legal `c_n_days_with_tx` CV 0.571 vs size 0.540 is a mid-busy bump (shape=flat), not a tail. Q6 CLOSE: lag-1 days 0.575 is lag0++0.004, not a lead. Fold 0 died because large groups there are already-neg (T3 Y2 17.3% vs rest-T3 4.3%) — not a short trail. Do not use family B as X. Trees stay PARK. Not a 0–100.

## Holdout coverage (LOW_POWER, not a claim)

Holdout labeled `y2_neg_2of3`: n=857 pos=23 cos=72 rate=0.027. LOW_POWER — not a KEEP claim. Quote train CV only.

## Parent return

- **Y4 crash overlap:** Y2pos crash=35.4%; Y2∩Y4=12.7% (cov 19%); Y4∩Y2=9.5%; Jaccard 0.057; pos in-ratio 1.03 vs Y4's 0.36. **Different turn. CLOSE, do not merge.**
- **Best legal single:** `c_n_days_with_tx` CV **0.571** vs size 0.540 / night GBM 0.540. Beats size by +0.03 but shape is flat (Q4 bump); body ≤17 CV 0.426. Drop 12 chronic names → days **0.549** vs size 0.544 (gap +0.005). The 0.571 *is* those names. No Q5 KEEP. No B-copy (max |ρ| 0.43).
- **Fold 0 inverted:** rate 11.1% vs rest 6.3%. Not late-trail. 12 chronic dark names in GROUP_0158/0172 hold **47%** of fold-0 positives. Drop the 4 T3 groups and fold-0 rate is 6.7% = rest; group_size CV 0.375 → 0.570. Trees stay PARK.
- **Q3/Q5 (no family B):** Y2 is already-negative persistence (82% of pos are below 0 at t; clean-now leftover 1.5%). The 82→68 *direction* is real; a non-B why is not. Dark 9.14% vs 6.34% confirmed — and all-dark minus GROUP_0158/0172 is **5.5%** (below invoiced). Not a new Y. Q6 CLOSE (lag1 days = lag0).

## Extra cuts

### 7 — honest 1-month Q6

Only lag-1 is an honest clock (Q6 quoted: longer leads die on the hidden 72). Never B.

| feature | lag | CV AUROC ± sd | train | sign | coverage | gap vs size |
|---|---:|---:|---:|---:|---:|---:|
| `a_io_ratio` | 0 | 0.473 ± 0.069 | 0.509 | +1 | 0.862 | -0.067 |
| `a_io_ratio_lag1` | 1 | 0.474 ± 0.065 | 0.509 | +1 | 0.794 | -0.066 |
| `c_n_days_with_tx` | 0 | 0.571 ± 0.046 | 0.577 | +1 | 1.000 | +0.031 |
| `c_n_days_with_tx_lag1` | 1 | 0.575 ± 0.046 | 0.580 | +1 | 0.931 | +0.035 |

Q6 KEEP lag-1 `c_n_days_with_tx` (must beat size by ≥0.02, stay <0.60, **and beat lag-0 by ≥0.01**): **False** (lag1 CV 0.575, lag_lift +0.004). Lag-1 ≈ lag-0 is the same busy-bin trait, not a lead. Do not claim Q6.

### 8 — fold 0 × so-far × group size

Company-month so-far (not company-total months-on-book). Fixed cuts from trail_length.md. Not a new histogram.

| fold | n | P(Y=1) | share already-neg | so-far<12 | so-far≥18 | median so-far |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3384 | 0.111 | 0.116 | 0.631 | 0.110 | 9.0 |
| 1 | 3158 | 0.028 | 0.039 | 0.666 | 0.114 | 8.0 |
| 2 | 3340 | 0.056 | 0.069 | 0.647 | 0.105 | 8.0 |
| 3 | 3787 | 0.082 | 0.094 | 0.607 | 0.137 | 9.0 |
| 4 | 3687 | 0.084 | 0.099 | 0.676 | 0.083 | 8.0 |

Fold 0 vs rest inside long so-far≥18: **0.105** vs 0.052 (gap +0.053, n_f0=372). Inside short<12: **0.111** vs 0.072 (gap +0.039). late_trail_effect=**False**  size_tercile_effect=**True**.

Y2 rate by so-far bucket × fold:

| bucket | fold | n | n_pos | P(Y=1) |
|---|---:|---:|---:|---:|
| <6 | 0 | 1120 | 115 | 0.103 |
| <6 | 1 | 1170 | 54 | 0.046 |
| <6 | 2 | 1140 | 110 | 0.096 |
| <6 | 3 | 1167 | 121 | 0.104 |
| <6 | 4 | 1282 | 123 | 0.096 |
| 6-11 | 0 | 1014 | 121 | 0.119 |
| 6-11 | 1 | 933 | 22 | 0.024 |
| 6-11 | 2 | 1020 | 39 | 0.038 |
| 6-11 | 3 | 1131 | 90 | 0.080 |
| 6-11 | 4 | 1209 | 90 | 0.074 |
| 12-17 | 0 | 878 | 99 | 0.113 |
| 12-17 | 1 | 694 | 9 | 0.013 |
| 12-17 | 2 | 828 | 27 | 0.033 |
| 12-17 | 3 | 971 | 65 | 0.067 |
| 12-17 | 4 | 891 | 67 | 0.075 |
| 18-23 | 0 | 372 | 39 | 0.105 |
| 18-23 | 1 | 361 | 4 | 0.011 |
| 18-23 | 2 | 352 | 11 | 0.031 |
| 18-23 | 3 | 518 | 34 | 0.066 |
| 18-23 | 4 | 305 | 31 | 0.102 |
| 24 | 0 | 0 | 0 | nan |
| 24 | 1 | 0 | 0 | nan |
| 24 | 2 | 0 | 0 | nan |
| 24 | 3 | 0 | 0 | nan |
| 24 | 4 | 0 | 0 | nan |

Y2 rate by group_size tercile (train cuts) × fold:

| T | interval | fold | n | n_pos | P(Y=1) |
|---:|---|---:|---:|---:|---:|
| 1 | (0.999, 7.0] | 0 | 1349 | 100 | 0.074 |
| 1 | (0.999, 7.0] | 1 | 1321 | 65 | 0.049 |
| 1 | (0.999, 7.0] | 2 | 1015 | 65 | 0.064 |
| 1 | (0.999, 7.0] | 3 | 1358 | 126 | 0.093 |
| 1 | (0.999, 7.0] | 4 | 1343 | 150 | 0.112 |
| 2 | (7.0, 13.0] | 0 | 636 | 32 | 0.050 |
| 2 | (7.0, 13.0] | 1 | 991 | 22 | 0.022 |
| 2 | (7.0, 13.0] | 2 | 1870 | 120 | 0.064 |
| 2 | (7.0, 13.0] | 3 | 1472 | 122 | 0.083 |
| 2 | (7.0, 13.0] | 4 | 523 | 53 | 0.101 |
| 3 | (13.0, 22.0] | 0 | 1399 | 242 | 0.173 |
| 3 | (13.0, 22.0] | 1 | 846 | 2 | 0.002 |
| 3 | (13.0, 22.0] | 2 | 455 | 2 | 0.004 |
| 3 | (13.0, 22.0] | 3 | 957 | 62 | 0.065 |
| 3 | (13.0, 22.0] | 4 | 1821 | 108 | 0.059 |

### 9 — already-neg vs clean-now (B decomp, never X)

If most Y2 positives are already `b_below_0`, the accepted label is persistence of the cash path, not an onset. That is why `y2_onset_neg` was rejected. Not X.

| slice | n | n_pos | companies | P(Y=1) |
|---|---:|---:|---:|---:|
| already_neg | 1465 | 1039 | 220 | 0.709 |
| clean_now | 15891 | 232 | 1165 | 0.015 |
| below_unknown | 0 | 0 | 0 | nan |

Persistence: **True**. Already-neg share of positives = 81.7%.

Legal singles on the clean-now leftover (honest turn, never B):

| feature | CV AUROC ± sd | n | n_pos | coverage |
|---|---:|---:|---:|---:|
| `c_n_days_with_tx` | 0.612 ± 0.073 | 15891 | 232 | 1.000 |
| `a_io_ratio` | 0.552 ± 0.079 | 15891 | 232 | 0.865 |
| `a_out6` | 0.582 ± 0.043 | 15891 | 232 | 0.667 |
| `log1p_a_in3` | 0.570 ± 0.059 | 15891 | 232 | 0.865 |
| `f_ds_r` | 0.628 ± 0.094 | 15891 | 232 | 0.865 |

### 10 — `c_n_days_with_tx` honesty

days>17: P(Y=1)=**10.0%** (n=6892, 691 pos) vs rest **5.5%**. Flag AUROC 0.579. Body days≤17 CV **0.426 ± 0.063**. Size inside busy bin 0.470 / rest 0.502. `d_cust_hhi` sign vs Y2 = -1 (Y4 was +1 — monopoly is the other turn).

### 11 — fold 0 large-group T3 × already-neg

fold0 T3 Y2=0.173 already-neg=0.164 vs rest T3 Y2=0.043 already-neg=0.050. Large groups are stressed in fold 0 and almost clean in folds 1–2 — group_size gets the wrong sign. That is the 0.344, not a short trail.

| slice | n | n_pos | cos | groups | P(Y=1) | share already-neg |
|---|---:|---:|---:|---:|---:|---:|
| f0_T3 | 1399 | 242 | 78 | 4 | 0.173 | 0.164 |
| rest_T3 | 4079 | 174 | 250 | 14 | 0.043 | 0.050 |
| f0_notT3 | 1985 | 132 | 148 | 43 | 0.066 | 0.081 |
| rest_notT3 | 9893 | 723 | 719 | 174 | 0.073 | 0.088 |

group_size diagnostic (never B, not a card):

| feature | CV AUROC ± sd | sign | folds |
|---|---:|---:|---|
| `group_size` | 0.375 ± 0.074 | +1 | 0.301 0.316 0.416 0.481 0.363 |
| `h_group_size` | 0.375 ± 0.074 | +1 | 0.301 0.316 0.416 0.481 0.363 |

### 12 — clean-now leftover (1.46%, not a new Y)

n=15891 pos=232. Below the 5% acceptance floor. leftover `f_ds_r` KEEP? **False**.

`f_ds_r` leftover quintiles shape=monotone_up:

| Q | n | n_pos | P(Y=1) | median |
|---:|---:|---:|---:|---:|
| 1 | 10999 | 120 | 0.011 | 0 |
| 2 | 2750 | 75 | 0.027 | 0.1369 |

`c_n_days_with_tx` leftover quintiles shape=flat:

| Q | n | n_pos | P(Y=1) | median |
|---:|---:|---:|---:|---:|
| 1 | 3225 | 31 | 0.010 | 2 |
| 2 | 3606 | 30 | 0.008 | 8 |
| 3 | 2947 | 31 | 0.011 | 15 |
| 4 | 3260 | 70 | 0.021 | 20 |
| 5 | 2853 | 70 | 0.025 | 27 |

`a_out6` leftover quintiles shape=flat:

| Q | n | n_pos | P(Y=1) | median |
|---:|---:|---:|---:|---:|
| 1 | 2121 | 25 | 0.012 | 3870 |
| 2 | 2120 | 17 | 0.008 | 1.437e+05 |
| 3 | 2120 | 21 | 0.010 | 6.471e+05 |
| 4 | 2120 | 33 | 0.016 | 2.174e+06 |
| 5 | 2120 | 47 | 0.022 | 1.217e+07 |

| feature | leftover CV ± sd | folds |
|---|---:|---|
| `f_ds_r` | 0.628 ± 0.094 | 0.595 0.785 0.532 0.619 0.609 |
| `c_n_days_with_tx` | 0.612 ± 0.073 | 0.582 0.533 0.689 0.565 0.688 |
| `a_out6` | 0.582 ± 0.043 | 0.564 0.603 0.535 0.560 0.645 |
| `a_io_ratio` | 0.552 ± 0.079 | 0.448 0.668 0.534 0.568 0.541 |

### 13 — the 4 fold-0 large groups

Four groups in fold 0 sit in the large tercile. That is the 0.344, not late-trail.

| group | n | n_pos | cos | P(Y=1) | already-neg | group_size | median so-far | late | mean days | dark share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `GROUP_0023` | 255 | 23 | 17 | 0.090 | 0.086 | 17 | 8 | 1.00 | 13.9 | 0.10 |
| `GROUP_0158` | 441 | 152 | 21 | 0.345 | 0.331 | 21 | 11 | 0.00 | 14.8 | 1.00 |
| `GROUP_0172` | 304 | 67 | 21 | 0.220 | 0.204 | 21 | 8 | 1.00 | 23.8 | 1.00 |
| `GROUP_0250` | 399 | 0 | 19 | 0.000 | 0.000 | 19 | 11 | 0.00 | 6.2 | 0.00 |

`GROUP_0158` holds **41%** of fold-0 positives (152/374). 0158 and 0172 are **all-dark** 21-company groups. 0158 is not late (late=0). 0250 is a large invoiced group at 0% Y2. Dark 9.14% is these clusters — not a new Y.

### 14 — days>17 × already-neg

Already-neg share busy=0.113 vs quiet=0.066. days CV on already-neg months 0.538 ± 0.081.

| slice | n | n_pos | P(Y=1) |
|---|---:|---:|---:|
| busy_already | 779 | 551 | 0.707 |
| busy_clean | 6113 | 140 | 0.023 |
| quiet_already | 686 | 488 | 0.711 |
| quiet_clean | 9778 | 92 | 0.009 |

### 16 — drop the 4 fold-0 T3 groups

Diagnostic mask only. Does not change the accepted Y or the holdout.

| slice | n | n_pos | P(Y=1) |
|---|---:|---:|---:|
| all_train | 17356 | 1271 | 0.073 |
| drop_4 | 15957 | 1029 | 0.064 |
| fold0_drop_4 | 1985 | 132 | 0.066 |
| fold0_keep_4 | 1399 | 242 | 0.173 |

| feature | CV after drop ± sd | folds |
|---|---:|---|
| `c_n_days_with_tx` | 0.555 ± 0.036 | 0.541 0.539 0.612 0.565 0.517 |
| `a_out6` | 0.552 ± 0.087 | 0.568 0.680 0.566 0.447 0.501 |
| `group_size` | 0.570 ± 0.101 | 0.427 0.684 0.584 0.519 0.637 |
| `log1p_a_in3` | 0.553 ± 0.045 | 0.534 0.609 0.594 0.519 0.511 |

### 17 — 360/110 + Y2 acf1

Pooled acf1 Spearman **+0.880** (n_pairs=16161); company-median **+0.686** (n=122). High persistence matches the already-neg story. Not a new Y. all-dark minus GROUP_0158/0172 is **5.5%** (below invoiced 6.3%) — the 9.14% dark lift *is* those two groups.

| slice | n | n_pos | cos | P(Y=1) |
|---|---:|---:|---:|---:|
| all_dark_360 | 4489 | 426 | 352 | 0.095 |
| mixed_110 | 1592 | 130 | 107 | 0.082 |
| invoiced | 11275 | 715 | 736 | 0.063 |
| all_dark_360_wo_0158_0172 | 3744 | 207 | 310 | 0.055 |

### 18 — already-neg inside 0158/0172 vs rest

If already-neg Y2 is ~70% both in-hot and outside, fold-0 death is *more already-neg months in two dark groups*, not a new process.

Same process? **False**.

| slice | n | n_pos | cos | P(Y=1) | share below 0 |
|---|---:|---:|---:|---:|---:|
| already_neg_in_0158_0172 | 208 | 177 | 19 | 0.851 | 1.000 |
| already_neg_outside | 1257 | 862 | 201 | 0.686 | 1.000 |
| clean_in_0158_0172 | 537 | 42 | 39 | 0.078 | 0.000 |
| clean_outside | 15354 | 190 | 1126 | 0.012 | 0.000 |
| 0158_0172_all | 745 | 219 | 42 | 0.294 | 0.279 |
| rest_all | 16611 | 1052 | 1153 | 0.063 | 0.076 |

### 19 — 0158 vs 0172 (already-neg / clean onset)

Both all-dark. 0158 is stickier already-neg *and* has leftover onset. 0250 is the large invoiced 0% control. Not a renamed Y.

| group | n | P(Y=1) | already n / rate | clean n / rate | below0 | cos |
|---|---:|---:|---|---|---:|---:|
| `GROUP_0158` | 441 | 0.345 | 146 / 0.842 | 295 / 0.098 | 0.331 | 21 |
| `GROUP_0172` | 304 | 0.220 | 62 / 0.871 | 242 / 0.054 | 0.204 | 21 |
| `GROUP_0023` | 255 | 0.090 | 22 / 0.636 | 233 / 0.039 | 0.086 | 17 |
| `GROUP_0250` | 399 | 0.000 | 0 / nan | 399 / 0.000 | 0.000 | 19 |

### 20 — holdout of the 4 groups + 0158 clean on days>0

Holdout coverage only (LOW_POWER). Not a claim.

| group | hold n | hold pos | hold cos | hold rate |
|---|---:|---:|---:|---:|
| `GROUP_0158` | 0 | 0 | 0 | nan |
| `GROUP_0172` | 0 | 0 | 0 | nan |
| `GROUP_0023` | 0 | 0 | 0 | nan |
| `GROUP_0250` | 0 | 0 | 0 | nan |

0158 clean + days>0: n=289 pos=28 rate=**9.7%**. Onset is not empty-month noise. None of the 4 groups appear in holdout labeled months.

### 21 — 0158 clean-now by so-far

If onset is only <6, it is a short-book artifact. If it stays high later, 0158 turns while already on book.

| so-far | n | n_pos | P(Y=1) |
|---|---:|---:|---:|
| <6 | 80 | 5 | 0.062 |
| 6-11 | 89 | 6 | 0.067 |
| 12-17 | 77 | 12 | 0.156 |
| 18+ | 49 | 6 | 0.122 |

### 22 — 0172 clean so-far + 0158 already-neg so-far

Compare 0158's late onset to 0172. 0158 already-neg by so-far shows when the pile sits below 0.

| slice | so-far | n | n_pos | P(Y=1) | below0 |
|---|---|---:|---:|---:|---:|
| 0172_clean | <6 | 95 | 8 | 0.084 | 0.000 |
| 0172_clean | 6-11 | 91 | 4 | 0.044 | 0.000 |
| 0172_clean | 12-17 | 56 | 1 | 0.018 | 0.000 |
| 0172_clean | 18+ | 0 | 0 | nan | nan |
| 0158_already | <6 | 25 | 23 | 0.920 | 1.000 |
| 0158_already | 6-11 | 37 | 31 | 0.838 | 1.000 |
| 0158_already | 12-17 | 49 | 42 | 0.857 | 1.000 |
| 0158_already | 18+ | 35 | 27 | 0.771 | 1.000 |
| 0158_all | <6 | 105 | 28 | 0.267 | 0.238 |
| 0158_all | 6-11 | 126 | 37 | 0.294 | 0.294 |
| 0158_all | 12-17 | 126 | 54 | 0.429 | 0.389 |
| 0158_all | 18+ | 84 | 33 | 0.393 | 0.417 |

### 23 — 0158 Y2pos inflow-crash share (label story, not X)

If 0158 late positives sit at in-ratio ~0.36, that cluster is Y4-like even though panel Y2 is not. Future A only.

| slice | n | crash defined | crash share | median in-ratio |
|---|---:|---:|---:|---:|
| Y2pos_0158 | 152 | 141 | 0.333 | 1.11 |
| Y2pos_0172 | 67 | 63 | 0.254 | 1.02 |
| Y2pos_hot | 219 | 204 | 0.309 | 1.07 |
| Y2pos_rest | 1052 | 767 | 0.366 | 1.02 |
| Y2pos_0158_sofar_ge12 | 87 | 86 | 0.384 | 0.97 |

### 24 — Y2pos × Y4 / Y9 in 0158/0172 vs rest

If the cluster's Y2 pos were mostly Y4, fold-0 would be Y4's crash wearing a group mask. Y9 is a fee check only.

| slice | n | Y4 defined / share | Y9 defined / share |
|---|---:|---|---|
| Y2pos_0158 | 152 | 42 / 0.167 | 111 / 0.126 |
| Y2pos_0172 | 67 | 36 / 0.000 | 39 / 0.385 |
| Y2pos_hot | 219 | 78 / 0.090 | 150 / 0.193 |
| Y2pos_rest | 1052 | 167 / 0.144 | 454 / 0.139 |

### 25 — legal-X medians in 0158 / 0172 / rest

Days CV after dropping only 0158+0172: **0.557** ± 0.035. If this falls toward size 0.54, the 0.571 bump was the cluster.

| slice | n | days | a_io | a_out6 | a_in3 | d_cust_hhi | f_ds_r |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0158 | 441 | 17.00 | 0.934 | 6.105e+05 | 3.167e+05 | nan | 0 |
| 0172 | 304 | 23.00 | 0.0586 | 2.073e+07 | 3.167e+05 | nan | 0 |
| rest | 16611 | 14.00 | 0.983 | 6.446e+05 | 2.658e+05 | 0.485 | 0 |

### 26 — singles on rest (drop 0158+0172)

Best legal on rest: `d_cust_hhi` CV **0.564** vs size 0.555 gap=+0.009. KEEP on rest would still need tail/monotone + ≥0.02. None expected.

| feature | CV ± sd | coverage | folds |
|---|---:|---:|---|
| `c_n_days_with_tx` | 0.557 ± 0.035 | 1.000 | 0.552 0.539 0.612 0.565 0.517 |
| `a_out6` | 0.555 ± 0.088 | 0.659 | 0.584 0.680 0.566 0.447 0.501 |
| `a_io_ratio` | 0.477 ± 0.067 | 0.861 | 0.461 0.571 0.481 0.384 0.490 |
| `d_cust_hhi` | 0.564 ± 0.161 | 0.429 | 0.707 0.699 0.325 0.600 0.488 |
| `f_ds_r` | 0.518 ± 0.078 | 0.861 | 0.442 0.621 0.438 0.532 0.557 |
| `c_zero_in_month` | 0.520 ± 0.027 | 1.000 | 0.516 0.539 0.555 0.504 0.486 |
| `log1p_a_in3` | 0.555 ± 0.044 | 0.861 | 0.540 0.609 0.594 0.519 0.511 |

### 27 — HHI fold-2 inversion (rest; 0158/0172 have no HHI)

HHI>0.975 is protective on Y2 (Y4's tail). Fold-2 inversion is a second cluster, not a Q5 KEEP.

| slice | n | n_pos | cos | groups | P(Y=1) | med HHI |
|---|---:|---:|---:|---:|---:|---:|
| rest_hhi_def | 7130 | 445 | 651 | 147 | 0.062 | 0.485 |
| rest_hhi_tail | 1479 | 67 | 283 | 101 | 0.045 | 1.000 |
| fold2_hhi_def | 1438 | 79 | 139 | 28 | 0.055 | 0.333 |
| fold2_hhi_tail | 188 | 24 | 47 | 19 | 0.128 | 1.000 |
| not_f2_hhi_tail | 1291 | 43 | 236 | 82 | 0.033 | 1.000 |

Fold-2 HHI-defined groups by n_pos:

- `GROUP_0013` n=178 pos=14 rate=0.079
- `GROUP_0094` n=176 pos=14 rate=0.080
- `GROUP_0230` n=57 pos=11 rate=0.193
- `GROUP_0088` n=75 pos=10 rate=0.133
- `GROUP_0241` n=29 pos=10 rate=0.345

### 28 — fold-2 HHI-hot groups (footnote, not a card)

HHI tail is protective on Y2 except fold 2. These groups do not rewrite Q5.

| group | n | P(Y=1) | cos | below0 | med days | med HHI | dark | fold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `GROUP_0241` | 35 | 0.457 | 7 | 0.571 | 13.0 | 0.771 | 0.00 | 2 |
| `GROUP_0230` | 63 | 0.190 | 7 | 0.206 | 10.0 | 0.799 | 0.00 | 2 |
| `GROUP_0088` | 85 | 0.129 | 5 | 0.153 | 11.0 | 0.238 | 0.00 | 2 |
| `GROUP_0013` | 226 | 0.080 | 12 | 0.102 | 11.0 | 0.161 | 0.00 | 2 |
| `GROUP_0094` | 255 | 0.063 | 13 | 0.071 | 18.0 | 0.103 | 0.00 | 2 |

### 29 — GROUP_0241 (tiny invoiced pile, not a card)

n=35 pos=16 rate=45.7% cos=7. Already-neg 20 at 80.0%; clean 15 at 0.0%. Crash share 33.3% med in-ratio 3.19 (not Y4's 0.36). Same persistence, 7 companies. Not Q5.

### 30 — 0158 per-company already-neg share

21 companies: 14 ever Y2-pos; 6 spend ≥50% of labeled months below 0; 2 always-below; 8 never-below. Median share-below **19%**, median company Y2 **19%**. A handful of chronic names, not the whole 21.

trail_length.md: Y2 short so-far 7.9% vs long 6.2% — fold-0 11.1% is not that short-book pile. Histogram not redone.

### 31 — fold-0 positives in the 0158 chronic names

Fold-0 Y2 pos=374. In 0158: 152 (41%). In the 6 chronic (≥50% months below 0) names: 110 (29%). A handful of names, not a group law.

### 32 — 0172 chronic + combined fold-0 share

0172: 21 cos, 6 ever Y2, 6 chronic, 15 never-below, median share-below 0%. 0158+0172 chronic (12 names) hold **47%** of fold-0 positives (177/374).

### 33 — legal-X medians on the 12 chronic names

If days look like rest, the 0.571 single is not even these names.

| slice | n | n_pos | med days | med io | med in3 | med out6 |
|---|---:|---:|---:|---:|---:|---:|
| chronic_12 | 216 | 177 | 21.5 | 0.781 | 3.91e+05 | 3.39e+06 |
| chronic_12_pos | 177 | 177 | 22.0 | 0.778 | 3.911e+05 | 3.205e+06 |
| rest | 17140 | 1094 | 14.0 | 0.977 | 2.666e+05 | 6.545e+05 |

### 34 — drop the 12 chronic companies only

n=17140 pos=1094. Days CV **0.549** ± 0.041 vs size 0.544 gap=+0.005. Folds 0.513 0.539 0.612 0.565 0.517. If gap < 0.02 the 0.571 single *is* those 12 names.

### 35 — fold-0 rate without the 12 chronic names

| slice | n | n_pos | P(Y=1) |
|---|---:|---:|---:|
| fold0 | 3384 | 374 | 0.111 |
| fold0_wo_12 | 3168 | 197 | 0.062 |
| rest_wo_12 | 13972 | 897 | 0.064 |
| fold0_only_12 | 216 | 177 | 0.819 |


