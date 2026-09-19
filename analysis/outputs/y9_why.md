# Y9 why — mix shift or outflow tail

- **When:** 2026-09-19T02:24
- **Agent:** `0511f2af`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.evaluate.y9_why`
- **Holdout:** 72 companies, seed 20260918. Coverage only. Rates + singles on train.
- **Y:** `y9_fee_r_ownp80` / `y9_fee_spike` from `targets.parquet` (not rebuilt). Train own-p80 9591 / 1350 / **14.08%**; spike 7879 / 1502 / **19.06%**.
- **X candidates:** Family M in memory (`m_fee_share`, `m_int_share`; `m_fin_share` caution) + `a_out6` / `log1p(a_in3)` / `a_uncat_share`. **Never F. Never `a_fin_cost`. Never D/E.**
- **Quote:** train group-fold CV (5 folds). Holdout is LOW_POWER.
- **Brief:** Q3 turning / Q5 why. Q6 only via honest 1-month lag (t3 mix already CLOSE).
- Not bankruptcy. Not a 0–100.

## Decision

**CLOSE** Family M as a Y9 Q5 column. Y9 is **not an outflow tail** and **not a usable mix shift**. Merge: **no**.

NO — do not merge Family M. No m_* clears a_out6 + 0.02 without leak/size.

- Best raw `m_fin_share` CV **0.618** vs `a_out6` **0.565** — leak (ρ≥0.80 vs `a_fin_cost`). Legal leftover `m_int_share` 0.525 CLOSE.
- Mix that beats outflow equals any-`a_fin_cost>0` (0.610 vs `m_fee_share` 0.613). That is the Y. PARK. Never F / `a_fin_cost` as X.
- First-fin month: 47.5% own-p80 / 86.3% spike. 75% of own-p80 pos are later-fin. The 64 before-first-fin positives are **exactly** 1 month before onset. Not Q6.
- GBM PARK: `a_out6` 0.565 is a no-fee shield (Q1 no-fee 6.7% vs any-fee 22.9%); it dies on later-fin (0.51) and on large∩any-fee (0.530). Shield survives inside size terciles. Spike `a_op_in` 0.563 is size (0.94).
- later-fin fee-share is tail-only (15.8% → 22.5%; single 0.550). Fee vs int ρ=+0.11. Company-median fee-share acf1 −0.055.
- Label stays. Dark = invoiced 14.1%. Never D/E.

## Pass 1 — decompose positives

High flags are that company's own expanding p80 (months ≤ t, min 6 finite). Not a pooled cut. Y2 / Y4 are accepted labels from the same parquet.

| Y | flag | n_hi / n_defined pos | share of pos | coverage of pos |
|---|---|---:|---:|---:|
| `y9_fee_r_ownp80` | `a_out6_hi` | 288 / 926 | 0.311 | 0.686 |
| `y9_fee_r_ownp80` | `m_fee_share_hi` | 533 / 1329 | 0.401 | 0.984 |
| `y9_fee_r_ownp80` | `m_int_share_hi` | 200 / 1329 | 0.150 | 0.984 |
| `y9_fee_r_ownp80` | `a_uncat_share_hi` | 275 / 1329 | 0.207 | 0.984 |
| `y9_fee_r_ownp80` | `y2_neg_2of3` | 92 / 1349 | 0.068 | 0.999 |
| `y9_fee_r_ownp80` | `y4_ds_r_double` | 59 / 283 | 0.208 | 0.210 |
| `y9_fee_spike` | `a_out6_hi` | 348 / 1062 | 0.328 | 0.707 |
| `y9_fee_spike` | `m_fee_share_hi` | 618 / 1480 | 0.418 | 0.985 |
| `y9_fee_spike` | `m_int_share_hi` | 220 / 1480 | 0.149 | 0.985 |
| `y9_fee_spike` | `a_uncat_share_hi` | 298 / 1480 | 0.201 | 0.985 |
| `y9_fee_spike` | `y2_neg_2of3` | 87 / 1501 | 0.058 | 0.999 |
| `y9_fee_spike` | `y4_ds_r_double` | 80 / 299 | 0.268 | 0.199 |

Own-p80 positives that are high-`a_out6`: **31.1%**. High-`m_fee_share`: **40.1%**. Read: **mixed** — neither story dominates.

## Pass 2 — quintiles (train labeled cuts)

### `m_fee_share` vs `y9_fee_r_ownp80`

n=9134 bins=3 monotone=True tail_only=True

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 6.51e-05] | 5480 | 620 | 0.113 | 0 |
| 2 | (6.51e-05, 0.000572] | 1827 | 313 | 0.171 | 0.0002137 |
| 3 | (0.000572, 1.0] | 1827 | 403 | 0.221 | 0.002235 |

### `m_fee_share` vs `y9_fee_spike`

n=7663 bins=4 monotone=False tail_only=True

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 9.08e-06] | 3065 | 509 | 0.166 | 0 |
| 2 | (9.08e-06, 0.000145] | 1533 | 300 | 0.196 | 4.76e-05 |
| 3 | (0.000145, 0.000834] | 1532 | 278 | 0.181 | 0.0003407 |
| 4 | (0.000834, 1.0] | 1533 | 397 | 0.259 | 0.003186 |

### `m_int_share` vs `y9_fee_r_ownp80`

n=9134 bins=1 monotone=None tail_only=None

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 1.0] | 9134 | 1336 | 0.146 | 0 |

### `m_int_share` vs `y9_fee_spike`

n=7663 bins=2 monotone=True tail_only=None

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-0.001, 3.5e-06] | 6130 | 1182 | 0.193 | 0 |
| 2 | (3.5e-06, 1.0] | 1533 | 302 | 0.197 | 0.0002888 |

### `a_out6` vs `y9_fee_r_ownp80`

n=9591 bins=5 monotone=False tail_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-28157774.900999997, 42498.26] | 1919 | 155 | 0.081 | 3041 |
| 2 | (42498.26, 352286.02] | 1918 | 286 | 0.149 | 1.432e+05 |
| 3 | (352286.02, 1214223.85] | 1918 | 278 | 0.145 | 6.731e+05 |
| 4 | (1214223.85, 4294706.09] | 1918 | 299 | 0.156 | 2.238e+06 |
| 5 | (4294706.09, 17114407320.57] | 1918 | 332 | 0.173 | 1.232e+07 |

### `a_out6` vs `y9_fee_spike`

n=7879 bins=5 monotone=False tail_only=False

| Q | interval | n | n_pos | P(Y=1) | median X |
|---:|---|---:|---:|---:|---:|
| 1 | (-19463081.981, 111456.59] | 1576 | 274 | 0.174 | 1.579e+04 |
| 2 | (111456.59, 568031.532] | 1576 | 343 | 0.218 | 2.935e+05 |
| 3 | (568031.532, 1721514.658] | 1575 | 289 | 0.183 | 1.041e+06 |
| 4 | (1721514.658, 5528823.174] | 1576 | 310 | 0.197 | 2.901e+06 |
| 5 | (5528823.174, 17114407320.57] | 1576 | 286 | 0.181 | 1.475e+07 |

Plot: `y9_fee_share_quintiles.png`.

## Pass 3 — single-feature CV

| Y | feature | CV AUROC ± sd | train | sign | coverage | size AUROC | size ρ |
|---|---|---:|---:|---:|---:|---:|---:|
| `y9_fee_r_ownp80` | `m_fee_share` | 0.613 ± 0.023 | 0.610 | +1 | 0.952 | 0.589 | +0.201 |
| `y9_fee_r_ownp80` | `m_int_share` | 0.525 ± 0.019 | 0.528 | +1 | 0.952 | 0.546 | +0.119 |
| `y9_fee_r_ownp80` | `m_fin_share` | 0.618 ± 0.017 | 0.619 | +1 | 0.952 | 0.557 | +0.120 |
| `y9_fee_r_ownp80` | `m_fee_n_share` | 0.611 ± 0.022 | 0.611 | +1 | 0.952 | 0.589 | +0.198 |
| `y9_fee_r_ownp80` | `a_out6` | 0.565 ± 0.033 | 0.567 | +1 | 1.000 | 0.891 | +0.777 |
| `y9_fee_r_ownp80` | `a_out3` | 0.558 ± 0.033 | 0.558 | +1 | 1.000 | 0.900 | +0.800 |
| `y9_fee_r_ownp80` | `log1p_a_in3` | 0.534 ± 0.019 | 0.534 | +1 | 1.000 | 0.999 | +1.000 |
| `y9_fee_r_ownp80` | `a_op_in` | 0.521 ± 0.023 | 0.520 | +1 | 1.000 | 0.935 | +0.876 |
| `y9_fee_r_ownp80` | `a_uncat_share` | 0.520 ± 0.029 | 0.522 | -1 | 0.952 | 0.521 | +0.026 |
| `y9_fee_r_ownp80` | `m_uncat_share` | 0.511 ± 0.029 | 0.512 | -1 | 0.952 | 0.533 | +0.054 |
| `y9_fee_spike` | `m_fee_share` | 0.563 ± 0.034 | 0.559 | +1 | 0.973 | 0.528 | +0.079 |
| `y9_fee_spike` | `m_int_share` | 0.487 ± 0.025 | 0.505 | +1 | 0.973 | 0.536 | +0.074 |
| `y9_fee_spike` | `m_fin_share` | 0.578 ± 0.036 | 0.575 | +1 | 0.973 | 0.513 | -0.032 |
| `y9_fee_spike` | `m_fee_n_share` | 0.541 ± 0.019 | 0.541 | +1 | 0.973 | 0.524 | +0.075 |
| `y9_fee_spike` | `a_out6` | 0.494 ± 0.016 | 0.504 | -1 | 1.000 | 0.896 | +0.796 |
| `y9_fee_spike` | `a_out3` | 0.511 ± 0.012 | 0.511 | -1 | 1.000 | 0.908 | +0.822 |
| `y9_fee_spike` | `log1p_a_in3` | 0.518 ± 0.017 | 0.516 | -1 | 1.000 | 1.000 | +1.000 |
| `y9_fee_spike` | `a_op_in` | 0.563 ± 0.018 | 0.562 | -1 | 1.000 | 0.938 | +0.880 |
| `y9_fee_spike` | `a_uncat_share` | 0.494 ± 0.018 | 0.507 | +1 | 0.973 | 0.551 | +0.091 |
| `y9_fee_spike` | `m_uncat_share` | 0.489 ± 0.019 | 0.503 | +1 | 0.973 | 0.563 | +0.120 |

`a_out6` on `y9_fee_r_ownp80` is the GBM single to beat (**0.565**, night quote 0.565). KEEP a mix column only if it beats that by ≥0.02 and is not an F-copy / size proxy.

### Mix-column letters

| feature | decision | reason |
|---|---|---|
| `m_fee_share` | **PARK** | leak |ρ|≥0.8 vs F / a_fin_cost |
| `m_int_share` | **CLOSE** | ties or loses to outflow (CV 0.525 vs a_out6 0.565, gap -0.040) |
| `m_fin_share` | **PARK** | leak |ρ|≥0.8 vs F / a_fin_cost |
| `m_fee_n_share` | **PARK** | numeric gap vs a_out6 +0.046 but fee-presence rewrite (gap vs any-fin +0.001; fee>0 CV 0.545) |
| `m_uncat_share` | **CLOSE** | ties or loses to outflow (CV 0.511 vs a_out6 0.565, gap -0.054) |

## Pass 4 — leak vs F

Fail if Spearman |ρ| ≥ 0.8 vs `a_fin_cost` or `f_fc_r` (that is the Y).

| feature | vs | n | Spearman | Pearson | fail |
|---|---|---:|---:|---:|---|
| `m_fee_share` | `a_fin_cost` | 9134 | +0.830 | +0.014 | True |
| `m_fee_share` | `f_fc_r` | 9134 | +0.619 | +0.286 | False |
| `m_int_share` | `a_fin_cost` | 9134 | +0.415 | -0.003 | False |
| `m_int_share` | `f_fc_r` | 9134 | +0.326 | +0.398 | False |
| `m_fin_share` | `a_fin_cost` | 9134 | +0.875 | +0.006 | True |
| `m_fin_share` | `f_fc_r` | 9134 | +0.731 | +0.486 | False |
| `m_fee_n_share` | `a_fin_cost` | 9134 | +0.739 | +0.019 | False |
| `m_fee_n_share` | `f_fc_r` | 9134 | +0.531 | +0.221 | False |
| `m_uncat_share` | `a_fin_cost` | 9134 | +0.058 | +0.022 | False |
| `m_uncat_share` | `f_fc_r` | 9134 | -0.046 | +0.027 | False |
| `a_out6` | `a_fin_cost` | 9591 | +0.453 | +0.065 | False |
| `a_out6` | `f_fc_r` | 9591 | +0.142 | -0.013 | False |
| `a_uncat_share` | `a_fin_cost` | 9134 | -0.038 | +0.009 | False |
| `a_uncat_share` | `f_fc_r` | 9134 | -0.119 | -0.010 | False |

`m_fee_share` vs `m_int_share`: Spearman **+0.109**, Pearson +0.016. Not substitutes; fee and interest can be told apart.

## Pass 5 — dark 470 vs invoiced 744

Join-QA population (DuckDB book invoices). **Never D/E as X.**

| Y | slice | n | n_pos | companies | base rate |
|---|---|---:|---:|---:|---:|
| `y9_fee_r_ownp80` | train_all | 9591 | 1350 | 908 | 0.141 |
| `y9_fee_r_ownp80` | dark_470 | 3183 | 448 | 333 | 0.141 |
| `y9_fee_r_ownp80` | invoiced_744 | 6408 | 902 | 575 | 0.141 |
| `y9_fee_spike` | train_all | 7879 | 1502 | 787 | 0.191 |
| `y9_fee_spike` | dark_470 | 2488 | 471 | 274 | 0.189 |
| `y9_fee_spike` | invoiced_744 | 5391 | 1031 | 513 | 0.191 |

## Pass 6 — honest 1-month Q6

Only lag-1 is an honest clock (Q6 quoted: longer leads die on the hidden 72). Do not claim a fee lead from t3 mix.

| Y | feature | lag | CV AUROC ± sd | train | sign | coverage |
|---|---|---:|---:|---:|---:|---:|
| `y9_fee_r_ownp80` | `m_fee_share` | 0 | 0.613 ± 0.023 | 0.610 | +1 | 0.952 |
| `y9_fee_r_ownp80` | `m_fee_share_lag1` | 1 | 0.571 ± 0.041 | 0.570 | +1 | 0.957 |
| `y9_fee_r_ownp80` | `m_int_share` | 0 | 0.525 ± 0.019 | 0.528 | +1 | 0.952 |
| `y9_fee_r_ownp80` | `m_int_share_lag1` | 1 | 0.518 ± 0.021 | 0.522 | +1 | 0.957 |
| `y9_fee_r_ownp80` | `m_fee_n_share` | 0 | 0.611 ± 0.022 | 0.611 | +1 | 0.952 |
| `y9_fee_r_ownp80` | `m_fee_n_share_lag1` | 1 | 0.578 ± 0.033 | 0.578 | +1 | 0.957 |
| `y9_fee_spike` | `m_fee_share` | 0 | 0.563 ± 0.034 | 0.559 | +1 | 0.973 |
| `y9_fee_spike` | `m_fee_share_lag1` | 1 | 0.539 ± 0.042 | 0.543 | -1 | 0.977 |
| `y9_fee_spike` | `m_int_share` | 0 | 0.487 ± 0.025 | 0.505 | +1 | 0.973 |
| `y9_fee_spike` | `m_int_share_lag1` | 1 | 0.524 ± 0.023 | 0.521 | -1 | 0.977 |
| `y9_fee_spike` | `m_fee_n_share` | 0 | 0.541 ± 0.019 | 0.541 | +1 | 0.973 |
| `y9_fee_spike` | `m_fee_n_share_lag1` | 1 | 0.526 ± 0.022 | 0.526 | -1 | 0.977 |

## Mapping (Q3 / Q5 / Q6)

Y9 is who is *turning* on fee+interest / inflow (Q3, FinRegLab NSF/fee). The why (Q5) on train is **not an outflow tail** (31% of own-p80 positives are high `a_out6`; 2×2 neither is 42%). It is also **not a usable mix shift**: `m_fee_share` CV 0.613 looks like it beats `a_out6` 0.565, but that is fee-*presence* (any-`a_fin_cost` CV 0.610, gap +0.001; on fee>0 support mix drops to 0.55). `m_fee_share` / `m_fin_share` fail the |ρ|≥0.80 leak vs `a_fin_cost`. Fee and interest shares are not substitutes (ρ 0.11). Lag-1 fee-share 0.571 does not clear `a_out6`+0.02 — not Q6. The 64 before-first-fin own-p80 positives are exactly 1 month before onset (algebraic forward label). t3 mix already CLOSE. Verdict **CLOSE**. Merge: no. Never F. Not a 0–100.

## Parent return

- Y9 is **not an outflow tail** (31% of own-p80 pos high `a_out6`; 2×2 neither 42%). It is **not a usable mix shift**: columns that beat `a_out6` 0.565 are fee-presence rewrites of `a_fin_cost>0` (any-fin 0.610; `m_fee_share` 0.613).
- Best raw single `m_fin_share` **0.618** — leak ρ 0.875 vs `a_fin_cost`. Best legal `m_int_share` **0.525** (loses to outflow). Leak fail ≥0.80: `m_fee_share` 0.830, `m_fin_share` 0.875. `m_fee_n_share` 0.739 is a presence rewrite.
- Merge Family M: **no**. KEEP gate (beat `a_out6` by ≥0.02 and not F-copy) is not met.
- Label stays. GBM PARK explained (no-fee shield + size `a_op_in`). Not Q6 (lag1 0.571; company-median acf1 −0.055; 64 before-first-fin pos are exactly 1m before onset).

## Extra cuts

### 1b — 2×2 high-out × high-fee among own-p80 positives

n=914  out_only=158  fee_only=242  both=127  neither=387 (neither share **42.3%**).
Largest cell is *neither*. Y9 is not 'spent more', and own-p80 high-fee is not most of it.

### 7 — leak on the positive support

All-row Spearman vs `a_fin_cost` can be zeros lining up (no fee month ⇒ no fin_cost).

| feature | vs | n all | ρ all | n support | ρ support | fail all / support |
|---|---|---:|---:|---:|---:|---|
| `m_fee_share` | `a_fin_cost` | 9134 | +0.830 | 5177 | +0.501 | True / False |
| `m_fee_share` | `f_fc_r` | 9134 | +0.619 | 5177 | +0.678 | False / False |
| `m_int_share` | `a_fin_cost` | 9134 | +0.415 | 1640 | +0.008 | False / False |
| `m_int_share` | `f_fc_r` | 9134 | +0.326 | 1640 | +0.613 | False / False |
| `m_fin_share` | `a_fin_cost` | 9134 | +0.875 | 5663 | +0.508 | True / False |
| `m_fin_share` | `f_fc_r` | 9134 | +0.731 | 5663 | +0.741 | False / False |
| `m_fee_n_share` | `a_fin_cost` | 9134 | +0.739 | 5177 | +0.077 | False / False |
| `m_fee_n_share` | `f_fc_r` | 9134 | +0.531 | 5177 | +0.314 | False / False |

Any-fee vs any-`a_fin_cost`: AUROC **0.957**, agree 94.7%, fee-only 2, fin-only 486 (interest without a fee).

### 8 — any-fee vs intensity

| Y | slice | n | n_pos | P(Y=1) |
|---|---|---:|---:|---:|
| `y9_fee_r_ownp80` | fee_eq_0 | 3955 | 365 | 0.092 |
| `y9_fee_r_ownp80` | fee_gt_0 | 5179 | 971 | 0.187 |
| `y9_fee_spike` | fee_eq_0 | 2485 | 401 | 0.161 |
| `y9_fee_spike` | fee_gt_0 | 5178 | 1083 | 0.209 |

| Y | feature | slice | CV AUROC ± sd | n |
|---|---|---|---:|---:|
| `y9_fee_r_ownp80` | `_any_fee` | all_labeled | 0.593 ± 0.029 | 9134 |
| `y9_fee_r_ownp80` | `m_fee_share` | all_labeled | 0.613 ± 0.023 | 9134 |
| `y9_fee_r_ownp80` | `m_fee_share` | fee_gt_0 | 0.550 ± 0.030 | 5179 |
| `y9_fee_spike` | `_any_fee` | all_labeled | 0.535 ± 0.022 | 7663 |
| `y9_fee_spike` | `m_fee_share` | all_labeled | 0.563 ± 0.034 | 7663 |
| `y9_fee_spike` | `m_fee_share` | fee_gt_0 | 0.559 ± 0.036 | 5178 |

### 9 — mix inside outflow terciles

| Y | slice | n | n_pos | P(Y=1) | fee-share CV | a_out6 CV |
|---|---|---:|---:|---:|---:|---:|
| `y9_fee_r_ownp80` | out_T1 | 3197 | 334 | 0.104 | 0.628 | 0.588 |
| `y9_fee_r_ownp80` | out_T2 | 3197 | 484 | 0.151 | 0.624 | 0.480 |
| `y9_fee_r_ownp80` | out_T3 | 3197 | 532 | 0.166 | 0.537 | 0.522 |
| `y9_fee_r_ownp80` | not_high_out | 4875 | 638 | 0.131 | 0.633 | — |
| `y9_fee_spike` | out_T1 | 2626 | 524 | 0.200 | 0.601 | 0.568 |
| `y9_fee_spike` | out_T2 | 2626 | 497 | 0.189 | 0.543 | 0.503 |
| `y9_fee_spike` | out_T3 | 2627 | 481 | 0.183 | 0.521 | 0.465 |
| `y9_fee_spike` | not_high_out | 3964 | 714 | 0.180 | 0.572 | — |

### 10 — neither cell

n=387 of 914 2×2 own-p80 positives. Median own-p80 of `m_fee_share` among Y9 pos = **0.000148** (p90 0.003456; share of own-p80 that is 0: 21.3%). 'High fee-share' is often any crumb above a zero history.

| flag inside neither | n_hi / n | share |
|---|---:|---:|
| `m_int_share_hi` | 75 / 387 | 0.194 |
| `a_uncat_share_hi` | 78 / 387 | 0.202 |
| `y2_neg_2of3` | 22 / 386 | 0.057 |
| `y4_ds_r_double` | 17 / 92 | 0.185 |
| `any_fee` | 206 / 387 | 0.532 |
| `any_int` | 106 / 387 | 0.274 |

### 11 — 360 vs 110 (never D/E as X)

| Y | slice | n | n_pos | companies | base rate |
|---|---|---:|---:|---:|---:|
| `y9_fee_r_ownp80` | all_dark_360 | 2321 | 310 | 245 | 0.134 |
| `y9_fee_r_ownp80` | mixed_dark_110 | 862 | 138 | 88 | 0.160 |
| `y9_fee_spike` | all_dark_360 | 1792 | 299 | 200 | 0.167 |
| `y9_fee_spike` | mixed_dark_110 | 696 | 172 | 74 | 0.247 |

### 12 — own-p80 ∩ spike; holdout coverage

Train both-labeled 7879: both-pos 842, own-only 410, spike-only 660, Spearman +0.533. Two labels, not a rewrite.
Holdout mix coverage 97.3% (own-p80 labeled 392 / 73 pos). Holdout `m_fee_share` AUROC 0.615 (LOW_POWER, not a claim).

### 13 — `m_fee_n_share` honesty

Spearman vs amount `m_fee_share` **+0.885**. Any-`a_fin_cost` presence (comparator, never X) CV **0.610**. `m_fee_n_share` all-row CV **0.611** (gap vs presence +0.001). On fee>0 support CV **0.545**. presence_rewrite=True; intensity_dead=True.

| feature | slice | CV AUROC ± sd | n |
|---|---|---:|---:|
| `_any_fin` | all | 0.610 ± 0.023 | 9591 |
| `_any_fee` | all | 0.593 ± 0.029 | 9134 |
| `m_fee_n_share` | all | 0.611 ± 0.022 | 9134 |
| `m_fee_share` | all | 0.613 ± 0.023 | 9134 |
| `m_fee_n_share` | fee_gt_0 | 0.545 ± 0.026 | 5179 |
| `m_fee_share` | fee_gt_0 | 0.550 ± 0.030 | 5179 |

Honesty: the numeric KEEP on `m_fee_n_share` is fee-*presence*, not mix intensity. Same raw fee tickets that build `a_fin_cost`. **Do not merge.**

### 14 — presence clock and outflow × any-fee

Any-fin-cost now CV **0.610**; lag-1 **0.577**. Honest 1-month Q6 KEEP vs `a_out6`+0.02: **False**. Presence is contemporaneous (same as fee-share). Not a lead.

| a_out6 Q | no-fee n / P(Y=1) | any-fee n / P(Y=1) |
|---:|---:|---:|
| 1 | 1386 / 0.067 | 441 / 0.229 |
| 2 | 895 / 0.086 | 932 / 0.204 |
| 3 | 728 / 0.063 | 1098 / 0.198 |
| 4 | 525 / 0.143 | 1302 / 0.164 |
| 5 | 421 / 0.176 | 1406 / 0.178 |

Low-outflow Q1 is protective **only among no-fee months**. Any-fee months sit near the 14–22% Y9 rate in every outflow quintile. `a_out6` is a no-activity shield, not a spend-tail why.

Fold AUCs (train group-fold, own-p80):

- `a_out6` folds 0.607 / 0.545 / 0.563 / 0.587 / 0.526  cv=0.565
- `m_fee_share` folds 0.619 / 0.646 / 0.586 / 0.611 / 0.601  cv=0.613

### 15 — which fin_cost token

| Y | slice | n | n_pos | P(Y=1) |
|---|---|---:|---:|---:|
| `y9_fee_r_ownp80` | neither | 3469 | 279 | 0.080 |
| `y9_fee_r_ownp80` | fee_only | 4025 | 754 | 0.187 |
| `y9_fee_r_ownp80` | int_only | 486 | 86 | 0.177 |
| `y9_fee_r_ownp80` | both_tokens | 1154 | 217 | 0.188 |
| `y9_fee_spike` | neither | 1999 | 291 | 0.146 |
| `y9_fee_spike` | fee_only | 4024 | 870 | 0.216 |
| `y9_fee_spike` | int_only | 486 | 110 | 0.226 |
| `y9_fee_spike` | both_tokens | 1154 | 213 | 0.185 |

Interest-only months match fee-only on own-p80 (~18% vs neither 8%). The Y9 lump is **any fin_cost token present**, not fee vs interest intensity. `m_int_share` as a continuous share still loses (0.525) because interest is rare; the binary *presence* is the whole signal.

### 16 — 360 vs 110 × any-fin (never D/E)

| Y | slice | n | n_pos | P(Y=1) |
|---|---|---:|---:|---:|
| `y9_fee_r_ownp80` | all_dark_360_nofin | 1032 | 77 | 0.075 |
| `y9_fee_r_ownp80` | all_dark_360_anyfin | 1289 | 233 | 0.181 |
| `y9_fee_r_ownp80` | mixed_dark_110_nofin | 353 | 28 | 0.079 |
| `y9_fee_r_ownp80` | mixed_dark_110_anyfin | 509 | 110 | 0.216 |
| `y9_fee_spike` | all_dark_360_nofin | 503 | 65 | 0.129 |
| `y9_fee_spike` | all_dark_360_anyfin | 1289 | 234 | 0.182 |
| `y9_fee_spike` | mixed_dark_110_nofin | 187 | 40 | 0.214 |
| `y9_fee_spike` | mixed_dark_110_anyfin | 509 | 132 | 0.259 |

Own-p80 360 vs 110 is mostly presence (nofin ~7.5–7.9% both sides). Spike is not: mixed-dark nofin 21.4% vs all-dark nofin 12.9%. Do not put D/E on that gap. Not a merge reason.

### 17 — company-level trait vs turning

Spearman of each train company's any-fin month-share vs its Y9 month-share (≥3 labeled months). High ρ = always-fee firms, not turning.

| Y | companies | ever pos | ρ | fin-rate pos / neg cos | Y rate always-fin | Y rate never-fin |
|---|---:|---:|---:|---:|---:|---:|
| `y9_fee_r_ownp80` | 833 | 481 | +0.316 | 0.67 / 0.46 | 0.186 (n=293) | 0.003 (n=133) |
| `y9_fee_spike` | 715 | 489 | -0.026 | 0.68 / 0.71 | 0.173 (n=298) | 0.000 (n=26) |

Never-fin own-p80 ≈ 0 is algebraic (0 cannot exceed own p80; spike NaNs a zero base). Useful number: own-p80 ρ=+0.32 (mild company trait); spike ρ=−0.03 (turning, not a firm type). Label stays. Still cannot put presence in X.

### 18 — intensity on always-fin companies

Train companies with any-fin on ≥99% of own-p80 labeled months: 333 cos, 3307 months, 632 pos, rate 19.1%. Presence is saturated.

| Y | feature | CV AUROC ± sd | n |
|---|---|---:|---:|
| `y9_fee_r_ownp80` | `m_fee_share` | 0.544 ± 0.021 | 3307 |
| `y9_fee_r_ownp80` | `m_fee_n_share` | 0.548 ± 0.036 | 3307 |
| `y9_fee_r_ownp80` | `m_int_share` | 0.486 ± 0.012 | 3307 |
| `y9_fee_r_ownp80` | `a_out6` | 0.510 ± 0.034 | 3307 |
| `y9_fee_r_ownp80` | `log1p_a_in3` | 0.530 ± 0.016 | 3307 |
| `y9_fee_spike` | `m_fee_share` | 0.540 ± 0.059 | 3345 |
| `y9_fee_spike` | `m_fee_n_share` | 0.522 ± 0.029 | 3345 |
| `y9_fee_spike` | `m_int_share` | 0.491 ± 0.025 | 3345 |
| `y9_fee_spike` | `a_out6` | 0.545 ± 0.037 | 3345 |
| `y9_fee_spike` | `log1p_a_in3` | 0.546 ± 0.043 | 3345 |

Gap `m_fee_share` vs `a_out6` **+0.033**. KEEP intensity (must also clear 0.565+0.02, not just a dead `a_out6`)? **False**. `a_out6` collapses to ~0.51 once no-fee months are gone — that is the shield. Fee-share leftover 0.54 is not a Q5 card. Still do not merge.

### 19 — fee-share bins × outflow quintiles

Fee-share collapses (mostly zero). Bins are no-fee / fee-lo / fee-hi (median split on fee>0).

| Y | outflow bin | fee bin | n | n_pos | P(Y=1) |
|---|---|---|---:|---:|---:|
| `y9_fee_r_ownp80` | (-28157774.900999997, 61253.384] | no_fee | 1386 | 93 | 0.067 |
| `y9_fee_r_ownp80` | (-28157774.900999997, 61253.384] | fee_lo | 93 | 18 | 0.194 |
| `y9_fee_r_ownp80` | (-28157774.900999997, 61253.384] | fee_hi | 348 | 83 | 0.239 |
| `y9_fee_r_ownp80` | (61253.384, 407629.51] | no_fee | 895 | 77 | 0.086 |
| `y9_fee_r_ownp80` | (61253.384, 407629.51] | fee_lo | 315 | 60 | 0.190 |
| `y9_fee_r_ownp80` | (61253.384, 407629.51] | fee_hi | 617 | 130 | 0.211 |
| `y9_fee_r_ownp80` | (407629.51, 1310344.58] | no_fee | 728 | 46 | 0.063 |
| `y9_fee_r_ownp80` | (407629.51, 1310344.58] | fee_lo | 533 | 100 | 0.188 |
| `y9_fee_r_ownp80` | (407629.51, 1310344.58] | fee_hi | 565 | 117 | 0.207 |
| `y9_fee_r_ownp80` | (1310344.58, 4521613.552] | no_fee | 525 | 75 | 0.143 |
| `y9_fee_r_ownp80` | (1310344.58, 4521613.552] | fee_lo | 705 | 101 | 0.143 |
| `y9_fee_r_ownp80` | (1310344.58, 4521613.552] | fee_hi | 597 | 112 | 0.188 |
| `y9_fee_r_ownp80` | (4521613.552, 17114407320.57] | no_fee | 421 | 74 | 0.176 |
| `y9_fee_r_ownp80` | (4521613.552, 17114407320.57] | fee_lo | 944 | 159 | 0.168 |
| `y9_fee_r_ownp80` | (4521613.552, 17114407320.57] | fee_hi | 462 | 91 | 0.197 |

If fee-hi vs no-fee stays large in every outflow bin, leftover mix is still presence, not spend. Do not merge.

### 20 — first-fin onset vs later (never X)

Own-p80 onset − later_fin **+0.292**. Spike onset 0.863 / later_fin 0.201 / later_off 0.140.

| Y | slice | n | n_pos | P(Y=1) |
|---|---|---:|---:|---:|
| `y9_fee_r_ownp80` | onset_first_fin | 80 | 38 | 0.475 |
| `y9_fee_r_ownp80` | later_fin | 5583 | 1019 | 0.183 |
| `y9_fee_r_ownp80` | later_off | 2566 | 229 | 0.089 |
| `y9_fee_r_ownp80` | before_first_fin | 349 | 64 | 0.183 |
| `y9_fee_r_ownp80` | never_fin | 1013 | 0 | 0.000 |
| `y9_fee_spike` | onset_first_fin | 80 | 69 | 0.863 |
| `y9_fee_spike` | later_fin | 5583 | 1123 | 0.201 |
| `y9_fee_spike` | later_off | 2216 | 310 | 0.140 |
| `y9_fee_spike` | before_first_fin | 0 | 0 | nan |
| `y9_fee_spike` | never_fin | 0 | 0 | nan |

Onset vs later_fin tests turning-on vs presence-as-trait. later_off vs later_fin is the turning-off residual. Comparator only. Own-p80 pos mass: onset 38/1350, later_fin 1019/1350, before 64/1350.

### 21 — later_fin leftover and before-first-fin Q6

later_fin `m_fee_share` vs `a_out6` gap **+0.043**. KEEP_later (must clear 0.565+0.02 too)? **False**.

| Y | slice | feature | CV AUROC ± sd | n |
|---|---|---|---:|---:|
| `y9_fee_r_ownp80` | later_fin | `m_fee_share` | 0.550 ± 0.031 | 5583 |
| `y9_fee_r_ownp80` | later_fin | `m_int_share` | 0.480 ± 0.013 | 5583 |
| `y9_fee_r_ownp80` | later_fin | `a_out6` | 0.507 ± 0.021 | 5583 |
| `y9_fee_r_ownp80` | later_fin | `a_out3` | 0.522 ± 0.022 | 5583 |
| `y9_fee_r_ownp80` | later_fin | `log1p_a_in3` | 0.533 ± 0.021 | 5583 |
| `y9_fee_r_ownp80` | before_first_fin | `m_fee_share` | 0.500 ± 0.000 | 310 |
| `y9_fee_r_ownp80` | before_first_fin | `m_int_share` | 0.500 ± 0.000 | 310 |
| `y9_fee_r_ownp80` | before_first_fin | `a_out6` | 0.662 ± 0.112 | 349 |
| `y9_fee_r_ownp80` | before_first_fin | `a_out3` | 0.660 ± 0.091 | 349 |
| `y9_fee_r_ownp80` | before_first_fin | `log1p_a_in3` | 0.581 ± 0.086 | 349 |
| `y9_fee_spike` | later_fin | `m_fee_share` | 0.555 ± 0.036 | 5583 |
| `y9_fee_spike` | later_fin | `m_int_share` | 0.491 ± 0.032 | 5583 |
| `y9_fee_spike` | later_fin | `a_out6` | 0.543 ± 0.025 | 5583 |
| `y9_fee_spike` | later_fin | `a_out3` | 0.553 ± 0.027 | 5583 |
| `y9_fee_spike` | later_fin | `log1p_a_in3` | 0.553 ± 0.022 | 5583 |

before_first_fin best `a_out6` CV **0.662** ± 0.112 (folds 0.690 0.573 0.526 0.801 0.720; sizeAUC 0.600). KEEP_before **False**. n=349 / 64 pos is LOW_POWER; mix is 0.50. Do not claim a Q6 lead. Do not merge M.

### 22 — later_off leftover + before a_out6 quintiles

later_off best `a_out6` CV **0.579**. KEEP mix? **False**.

| Y | feature | CV AUROC ± sd | n |
|---|---|---:|---:|
| `y9_fee_r_ownp80` | `m_fee_share` | 0.500 ± 0.001 | 2301 |
| `y9_fee_r_ownp80` | `a_out6` | 0.579 ± 0.092 | 2566 |
| `y9_fee_r_ownp80` | `a_out3` | 0.577 ± 0.084 | 2566 |
| `y9_fee_r_ownp80` | `log1p_a_in3` | 0.554 ± 0.066 | 2566 |
| `y9_fee_spike` | `m_fee_share` | 0.502 ± 0.003 | 2000 |
| `y9_fee_spike` | `a_out6` | 0.581 ± 0.055 | 2216 |
| `y9_fee_spike` | `a_out3` | 0.584 ± 0.055 | 2216 |
| `y9_fee_spike` | `log1p_a_in3` | 0.566 ± 0.042 | 2216 |

before_first_fin `a_out6` quintiles n=349 bins=5 monotone=False tail_only=False (LOW_POWER).

| Q | n | n_pos | P(Y=1) | median a_out6 |
|---:|---:|---:|---:|---:|
| 1 | 71 | 8 | 0.113 | 142.5 |
| 2 | 69 | 6 | 0.087 | 1.069e+04 |
| 3 | 69 | 14 | 0.203 | 1.081e+05 |
| 4 | 70 | 16 | 0.229 | 3.7e+05 |
| 5 | 70 | 20 | 0.286 | 2.392e+06 |

later_off mix should be ~0.50. before quintiles stay LOW_POWER. Do not merge.

### 23 — months until first-fin (before slice, never X)

349 months / 67 companies. `months_to_first_fin` CV **1.000**. algebraic_clock=True.

| feature | CV AUROC ± sd | sign |
|---|---:|---:|
| `_months_to_first_fin` | 1.000 ± 0.000 | -1 |
| `_near_onset` | 0.862 ± 0.044 | +1 |

| months to first fin | n | n_pos | P(Y=1) |
|---|---:|---:|---:|
| 1m | 64 | 64 | 1.000 |
| 2-3m | 95 | 0 | 0.000 |
| 4-6m | 92 | 0 | 0.000 |
| 7m+ | 98 | 0 | 0.000 |

If the clock is strong, the 0.66 `a_out6` leftover is proximity to onset (Y looks at t+1..t+3), not a Q6 mix lead. Do not merge.

### 24 — later_fin fee-share quintiles (presence on)

If this is flat, leftover mix intensity is dead. Single CV 0.550 already loses the KEEP bar.

`m_fee_share` vs `y9_fee_r_ownp80` n=5583 bins=5 monotone=False tail_only=True

| Q | n | n_pos | P(Y=1) | median fee-share |
|---:|---:|---:|---:|---:|
| 1 | 1117 | 177 | 0.158 | 5.748e-07 |
| 2 | 1116 | 195 | 0.175 | 3.905e-05 |
| 3 | 1117 | 194 | 0.174 | 0.0001975 |
| 4 | 1116 | 202 | 0.181 | 0.0006893 |
| 5 | 1117 | 251 | 0.225 | 0.005086 |

`m_fee_share` vs `y9_fee_spike` n=5583 bins=5 monotone=False tail_only=True

| Q | n | n_pos | P(Y=1) | median fee-share |
|---:|---:|---:|---:|---:|
| 1 | 1117 | 206 | 0.184 | 5.748e-07 |
| 2 | 1116 | 213 | 0.191 | 3.905e-05 |
| 3 | 1117 | 204 | 0.183 | 0.0001975 |
| 4 | 1116 | 192 | 0.172 | 0.0006893 |
| 5 | 1117 | 308 | 0.276 | 0.005086 |

Tail-only (+4–9 pp in Q5). Not monotone. Not a Q5 mix card. Still do not merge.

### 25 — last pre-fee vs earlier outflow (within company)

Paired companies 49: share last>earlier **61.2%**, median Δ `a_out6` 979.1. If this is a ramp, the 0.66 leftover is spend-before-onset, still not mix, still not Q6.

| slice | n | median a_out6 | p80 |
|---|---:|---:|---:|
| last_pre | 64 | 3.6e+05 | 2.967e+06 |
| earlier_pre | 285 | 7.764e+04 | 7.273e+05 |

Do not merge. Do not put this clock in X.

### 26 — mix persistence (acf1, train)

`m_fee_share` company-median acf1 **-0.055** (pooled +0.315 is zeros lining up). persist=False. Mix cannot answer Q6.

| feature | pooled acf1 | pairs | median company acf1 | share |acf|≥0.3 |
|---|---:|---:|---:|---:|
| `m_fee_share` | +0.315 | 18826 | -0.055 | 0.260 |
| `m_int_share` | +0.663 | 18826 | -0.091 | 0.292 |
| `a_out6` | +0.992 | 13878 | +0.813 | 0.937 |

t3 mix already CLOSE. Lag-1 single does not clear 0.565+0.02. Not Q6.

### 27 — outflow quintiles inside size terciles

Q1 shield inside size? **True**. `a_out6` size AUROC 0.891 — if the Q1 gap dies inside terciles, the 0.565 bar is size.

| size T | out Q | n | n_pos | P(Y=1) | any-fee share |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 640 | 37 | 0.058 | 0.113 |
| 1 | 2 | 639 | 56 | 0.088 | 0.188 |
| 1 | 3 | 639 | 101 | 0.158 | 0.341 |
| 1 | 4 | 639 | 91 | 0.142 | 0.499 |
| 1 | 5 | 640 | 106 | 0.166 | 0.430 |
| 2 | 1 | 640 | 63 | 0.098 | 0.394 |
| 2 | 2 | 639 | 98 | 0.153 | 0.549 |
| 2 | 3 | 639 | 99 | 0.155 | 0.637 |
| 2 | 4 | 639 | 94 | 0.147 | 0.649 |
| 2 | 5 | 640 | 126 | 0.197 | 0.647 |
| 3 | 1 | 640 | 77 | 0.120 | 0.550 |
| 3 | 2 | 639 | 91 | 0.142 | 0.756 |
| 3 | 3 | 639 | 89 | 0.139 | 0.739 |
| 3 | 4 | 639 | 105 | 0.164 | 0.787 |
| 3 | 5 | 640 | 117 | 0.183 | 0.822 |

| size T | Q1 rate | rest rate | gap |
|---:|---:|---:|---:|
| 1 | 0.058 | 0.138 | -0.081 |
| 2 | 0.098 | 0.163 | -0.065 |
| 3 | 0.120 | 0.157 | -0.037 |

Comparator only. Do not merge M. Do not treat `a_out6` as a clean Q5 why.

### 28 — large ∩ any-fee leftover

n=2246 pos=354 rate 15.8%. KEEP mix? **False**.

| feature | CV AUROC ± sd | n |
|---|---:|---:|
| `m_fee_share` | 0.569 ± 0.046 | 2246 |
| `m_int_share` | 0.519 ± 0.040 | 2246 |
| `a_out6` | 0.530 ± 0.047 | 2246 |
| `a_out3` | 0.522 ± 0.042 | 2246 |
| `log1p_a_in3` | 0.520 ± 0.032 | 2246 |

If both mix and `a_out6` are ~0.50–0.54 here, the night 0.565 bar is the no-fee / small-activity shield. Still do not merge.


