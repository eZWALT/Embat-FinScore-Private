# Y3 Family I lift (train group-fold CV only)

Holdout 72 companies never in a fit (seed `20260918`). Stressed / labeled
`y3_recover_cash_6m` only (5648 / 402 / 7.12%, 26.7% of train company-months).
Spec: **50 trees, `max_depth=3`, `num_leaves=8`** — quoted shallow-A
**0.752 / n_x=15**. Family I via `analysis.features.interactions.build` **in
memory**. Parquet was **not** rewritten (confirmed: 118 cols, no `i_*`).
I was **not** added to FAMILIES.

Y3 X never family B. Forbidden: `i_runway_x_hhi`, `i_runway_x_ar30`,
`i_runway_x_zeroin`, `i_below0_x_payroll`.

KEEP if CV ≥ 0.772 (0.752 + 0.02) with extra columns. CLOSE if within 0.02
of 0.752. PARK if worse than 0.752 by > 0.02. 400+ES is diagnostic only.

## Baseline

Reproduced `lgbm_y3_core_shap_d3_n50` **0.7520 ± 0.037, n_x=15**. Folds
0.689, 0.780, 0.769, 0.752, 0.770. Delta vs quoted 0.752: **0.000**. Safe to
compare. Re-checked after later edits: still 0.7520.

Stems + lags 1,3: `c_ss_month`, `c_salary_month`, `a_n_tx`, `f_ds_r`,
`c_n_days_with_tx`. Never `a_op_in` (SIZE).

## Screen (train stressed)

All eight legal `i_*` kept. No SIZE. No NZV. Max |ρ| vs `b_runway`:
Spearman **0.162**, Pearson **0.212** (fail ≥ 0.80).

| feature | cov | modal | ρ runway S / P | ρ a_in3 | ρ a_op_in | vs 15-col (ρ) |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `i_transfer_x_ss` | 1.00 | 0.617 | −0.052 / 0.005 | 0.061 | 0.038 | 0.20 `c_ss_month` |
| `i_transfer_x_salary` | 1.00 | 0.661 | −0.077 / 0.005 | 0.071 | 0.054 | 0.24 `c_salary_month` |
| `i_io_x_zeroin` | 0.98 | 0.880 | −0.018 / −0.212 | −0.184 | −0.275 | −0.42 `a_n_tx_lag3` |
| `i_dso_x_dsr` | 0.42 | 0.593 | −0.068 / −0.017 | 0.212 | 0.214 | **0.92 `f_ds_r`** |
| `i_gap_x_supphhi` | 0.51 | 0.034 | −0.011 / −0.154 | −0.421 | −0.478 | **−0.81 `c_n_days_lag1`** |
| `i_io_x_dsr` | 0.98 | 0.603 | −0.033 / −0.037 | 0.275 | 0.295 | **0.98 `f_ds_r`** |
| `i_miss_e` | 1.00 | 0.641 | −0.162 / −0.022 | 0.054 | 0.049 | −0.16 |
| `i_miss_d` | 1.00 | 0.513 | −0.115 / −0.027 | 0.025 | 0.029 | −0.14 |

## Planned table (50 / depth-3)

| spec | n_x | CV ± sd | folds | vs 0.752 |
| --- | ---: | --- | --- | ---: |
| **baseline** | 15 | **0.7520 ± 0.037** | 0.689, 0.780, 0.769, 0.752, 0.770 | 0.000 |
| **A** SHAP pairs | 17 | **0.7522 ± 0.031** | 0.699, 0.771, 0.772, 0.754, 0.764 | +0.000 |
| **B** all legal | 23 | **0.7524 ± 0.018** | 0.720, 0.761, 0.758, 0.764, 0.758 | +0.000 |
| add `i_dso_x_dsr` | 16 | 0.7542 ± 0.023 | 0.717, 0.775, 0.759, 0.748, 0.771 | +0.002 |
| add `i_io_x_dsr` | 16 | 0.7541 ± 0.033 | 0.699, 0.780, 0.770, 0.750, 0.772 | +0.002 |
| add `i_transfer_x_ss` | 16 | 0.7526 ± 0.031 | 0.699, 0.772, 0.773, 0.752, 0.766 | +0.001 |
| add `i_transfer_x_salary` | 16 | 0.7508 ± 0.034 | 0.692, 0.765, 0.773, 0.754, 0.770 | −0.001 |
| add `i_gap_x_supphhi` | 16 | 0.7483 ± 0.030 | 0.702, 0.782, 0.755, 0.744, 0.759 | −0.004 |
| add `i_io_x_zeroin` | 16 | 0.7466 ± 0.030 | 0.698, 0.769, 0.762, 0.739, 0.765 | −0.005 |
| add `i_miss_d` | 16 | 0.7468 ± 0.033 | 0.695, 0.777, 0.765, 0.736, 0.761 | −0.005 |
| add `i_miss_e` | 16 | 0.7443 ± 0.034 | 0.689, 0.780, 0.760, 0.743, 0.750 | −0.008 |
| B drop `i_dso_x_dsr` | 22 | 0.7456 ± 0.027 | 0.700, 0.771, 0.756, 0.745, 0.756 | −0.006 |
| B drop `i_miss_e` | 22 | 0.7530 ± 0.021 | 0.716, 0.761, 0.761, 0.763, 0.765 | +0.001 |
| A + lags 1,3 | 21 | 0.7518 ± 0.026 | 0.707, 0.763, 0.770, 0.756, 0.764 | 0.000 |
| A 400+ES | 17 | 0.7379 ± 0.032 | collapsed | −0.014 |

PR-AUC: baseline 0.204, A 0.203, B 0.200, C3 0.209.

Seed sweep (5 LGB seeds, folds frozen): baseline 0.750±0.001,
A 0.751±0.002, B 0.749±0.002, C3 0.759±0.002. C3 never reaches 0.772.

Brier (lower better): baseline 0.166, A 0.168, B 0.172, C3 0.168.
Company-boot Brier: A−base +0.002 (P worse=0.69); C3−base +0.004
(P worse=0.72). I makes probabilities worse.

Company-level AUROC (~20 companies/fold with both classes): baseline
0.547, A 0.545, B 0.575, C3 0.582. Noisy; not a KEEP.

## Follow-up (same spec)

| spec | n_x | CV ± sd | vs 0.752 |
| --- | ---: | --- | ---: |
| C = dso+io | 17 | 0.7549 ± 0.026 | +0.003 |
| **C3** = C + transfer×ss | 18 | **0.7615 ± 0.024** | +0.009 |
| drop `f_ds_r` only (no I) | 12 | 0.7525 ± — | +0.000 |
| swap `f_ds_r`→`i_dso` | 15 | 0.7593 ± 0.015 | +0.007 |
| swap + SHAP pairs | 17 | 0.7627 ± 0.007 | +0.011 |
| drop `f_ds_r` + C3 extras | 15 | 0.7632 ± 0.016 | +0.011 |
| drop `f_ds_r` + dso + both SHAP | 15 | 0.7650 ± 0.010 | +0.013 |
| i_* only (8 legal) | 8 | 0.7138 ± 0.031 | −0.038 |
| best 400+ES | 15 | 0.7274 ± 0.034 | −0.025 (collapsed) |

Twelve random legal triples: 0.746–0.755. C3 is the best triple.

## Why this is not a merge

- `i_io_x_dsr` ρ=0.980 vs `f_ds_r`. `i_dso_x_dsr` ρ=0.916 vs `f_ds_r`.
- On the 2376 rows where `i_dso` is defined, 15-col is 0.789 and +`i_dso`
  is 0.788. The product does not lift where it exists.
- Permuting `i_dso` on C3 returns CV to 0.754 ≈ baseline.
- C3 val-fold perm drop: `i_dso` 0.022, `i_transfer_x_ss` 0.009, `i_io` 0.003.
  SHAP moves mass from `f_ds_r` to `i_dso` (substitution).
- Fold 0 `f_ds_r` single-AUROC is 0.547; `i_dso` is 0.681. `i_gap` is 0.832
  on fold 0 and **0.590** on fold 2 (trap).
- Company bootstrap: A−base +0.002 (P=0.57); B−base **−0.002** (P=0.47);
  C3−base +0.006 (P=0.69); add-`i_dso` +0.004 (P=0.67); 400× C3 +0.006.
  Drop `f_ds_r` with no I: Δ 0.000 (fold 0 +0.032, others negative).
  C3 − drop-`f_ds_r` (200×/fold): **+0.0046**, sd 0.021, P>0=0.68.
  Fold 0 is **−0.019** (I worse than just dropping `f_ds_r`).
  C3 − A (150×/fold): **+0.0046**, P>0=0.70 (fold 2 −0.012).
- Depth-1 baseline is already 0.758; I adds ~0.002 there. More trees / deeper
  trees / other lr or colsample never reach 0.772.
- Train-only isotonic: Brier collapses to ~0.067 for all three; C3 iso
  AUROC 0.759 vs baseline 0.751. Calibration does not create a KEEP.

## Verdict

**CLOSE — do not merge I into `monthly.parquet`.**

A and B are flat at 0.752. Best add C3 is +0.0095. Best substitution is
+0.013. KEEP needs 0.772. Family I stays in-memory only.

Six questions: `i_dso_x_dsr` is Q5 (collections × debt service) but it
duplicates `f_ds_r` already in the 15-col engine. The SHAP pairs are Q5
why / turning and add nothing once payroll flags are in X. Not a 0–100 score.
