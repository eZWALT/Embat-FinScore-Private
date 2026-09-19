# Wave 4 — Family I lift on Y3 shallow-A 0.752 (`gbm_i_lift`)

Agent `88f17954`. In-memory `i_*` on `y3_recover_cash_6m` (45→65 / who is
improving). No product. No 0–100. Holdout 72 never in a fit. Quote **train
group-fold CV**. Family B never in X. Parquet not rewritten. I not added to
FAMILIES.

Re-run:

```bash
/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.gbm_i_lift --phase AB
```

## Files written

- `analysis/models/gbm_i_lift.py` (owned)
- `analysis/outputs/i_lift.md`
- this note
- rows appended to `analysis/experiments/registry.csv`

Did not edit `interactions.py`, `gbm_core.py`, `gbm_y1.py`, `xgb_y4.py`,
`y10_util.py`, `build_targets.py`, parquet, or `product/`. Did not commit.

## Setup

| item | choice |
|------|--------|
| Y | `y3_recover_cash_6m`, stressed/labeled only |
| Train | 5648 / 402 / 7.12% (26.7% of train CM) |
| Split | 5 `group_folds`, seed `20260918` |
| Spec | **50 trees, max_depth=3, num_leaves=8** (quoted shallow-A 0.752 / n_x=15) |
| Core stems | `c_ss_month`, `c_salary_month`, `a_n_tx`, `f_ds_r`, `c_n_days_with_tx` + lags 1,3 |
| I | `interactions.build` in memory. Forbidden: `i_runway_x_*`, `i_below0_x_payroll` |
| Screens | train-stressed; drop if \|ρ\| vs `b_runway` ≥ 0.80 or vs `a_in3`/`a_op_in` > 0.85 or NZV |

Baseline reproduced **0.7520 ± 0.037, n_x=15** (folds 0.689, 0.780, 0.769,
0.752, 0.770). Within 0.00 of the quote. Safe to compare.

All eight legal `i_*` passed SIZE / NZV / B-leak. Max \|ρ\| vs `b_runway`:
Spearman 0.162 / Pearson 0.212.

## Quote (train CV only)

| spec | n_x | CV AUROC | sd | folds | vs 0.752 |
|------|----:|---------:|---:|-------|----------|
| **baseline** | 15 | **0.7520** | 0.037 | 0.689, 0.780, 0.769, 0.752, 0.770 | −0.000 |
| **A** SHAP pairs | 17 | **0.7522** | 0.031 | 0.699, 0.771, 0.772, 0.754, 0.764 | +0.000 |
| **B** all legal | 23 | **0.7524** | 0.018 | 0.720, 0.761, 0.758, 0.764, 0.758 | +0.000 |
| add `i_dso_x_dsr` | 16 | 0.7542 | 0.023 | 0.717, 0.775, 0.759, 0.748, 0.771 | +0.002 (boot Δ +0.004) |
| add `i_io_x_dsr` | 16 | 0.7541 | 0.033 | 0.699, 0.780, 0.770, 0.750, 0.772 | +0.002 |
| add `i_transfer_x_ss` | 16 | 0.7526 | 0.031 | 0.699, 0.772, 0.773, 0.752, 0.766 | +0.001 |
| add `i_transfer_x_salary` | 16 | 0.7508 | 0.034 | 0.692, 0.765, 0.773, 0.754, 0.770 | −0.001 |
| add `i_gap_x_supphhi` | 16 | 0.7483 | 0.030 | 0.702, 0.782, 0.755, 0.744, 0.759 | −0.004 |
| add `i_io_x_zeroin` | 16 | 0.7466 | 0.030 | 0.698, 0.769, 0.762, 0.739, 0.765 | −0.005 |
| add `i_miss_d` | 16 | 0.7468 | 0.033 | 0.695, 0.777, 0.765, 0.736, 0.761 | −0.005 |
| add `i_miss_e` | 16 | 0.7443 | 0.034 | 0.689, 0.780, 0.760, 0.743, 0.750 | −0.008 |
| B drop `i_dso_x_dsr` | 22 | 0.7456 | 0.027 | 0.700, 0.771, 0.756, 0.745, 0.756 | −0.006 |
| B drop `i_miss_e` | 22 | 0.7530 | 0.021 | 0.716, 0.761, 0.761, 0.763, 0.765 | +0.001 |
| **C3** (dso+io+ss) | 18 | **0.7615** | 0.024 | 0.721, 0.777, 0.763, 0.766, 0.781 | +0.009 |
| swap `f_ds_r`→`i_dso` + SHAP | 17 | 0.7627 | 0.007 | 0.757, 0.758, 0.766, 0.758, 0.774 | +0.011 |
| drop `f_ds_r` + dso + both SHAP | 15 | 0.7650 | 0.010 | 0.758, 0.761, 0.763, 0.761, 0.782 | +0.013 |
| A 400+ES | 17 | 0.7379 | 0.032 | collapsed 1–few trees | −0.014 |
| best 400+ES | 15 | 0.7274 | 0.034 | trees 1,6,1,1,3 | −0.025 |

## KEEP / PARK / CLOSE

Rule: KEEP if CV ≥ 0.772 with extra columns. CLOSE if within 0.02 of 0.752.
PARK if worse than 0.752 by > 0.02.

- A = 0.7522: no lift. Bootstrap A−baseline Δ = +0.002, P(Δ>0)=0.57.
  C3 − A (150×/fold) = +0.0046, P>0=0.70 (fold 2 −0.012).
- B = 0.7524: no lift (sd shrinks because fold 0 ↑ and fold 1 ↓).
  Bootstrap B−baseline Δ = **−0.002**, P(Δ>0)=0.47.
- Best add-on C3 = 0.7615 (+0.0095). Best substitution = 0.7650 (+0.013).
- Neither clears 0.772. Both inside the 0.02 band.
- Dropping fold 0 (the only fold I “fixes”) leaves C3 at 0.772 vs
  baseline 0.768 — **+0.004**, still not a merge. Do not cherry-pick.
- PR-AUC: baseline 0.204, A 0.203, B 0.200, C3 0.209. No PR lift.
- 400+ES collapses. Do not KEEP those numbers.

**Decision: CLOSE — do not merge I into parquet.** Parent should leave
Family I in-memory only.

## What this means (six questions)

Quiet-stressed recovery (Q2 who is improving; Q5 why) already lives in the
15-col depth-3 engine. Legal `i_*` are mostly recodes: `i_io_x_dsr` ρ=0.980
vs `f_ds_r`, `i_dso_x_dsr` ρ=0.916 vs `f_ds_r`, `i_gap` ρ=0.810 vs
`c_n_days_with_tx_lag1`. The Y3 SHAP pairs (`i_transfer_x_ss` /
`i_transfer_x_salary`) are weak singles (0.56) once `c_ss_month` /
`c_salary_month` are in X. Permuting `i_dso_x_dsr` on C3 returns CV to 0.754
≈ baseline. Depth-1 baseline is already 0.758; I adds ~0.002 there. The
depth-3 “+0.01” is mostly fixing fold 0: `f_ds_r` single-AUROC there is
0.547, `i_dso_x_dsr` is 0.681. `i_gap_x_supphhi` is 0.832 on fold 0 and
0.590 on fold 2 — do not add it.

Five LGB seeds (folds frozen): baseline 0.750, A 0.751, B 0.749, C3 0.759
(max 0.761). Brier: baseline 0.166 vs A 0.168 / B 0.172 / C3 0.168.
Company-boot Brier A−base +0.002 (P worse=0.69), C3−base +0.004
(P worse=0.72). Company-grain AUROC (~20 cos/fold): 0.547 / 0.545 /
0.575 / 0.582. I does not improve probabilities or company ranking
enough to merge.

Company bootstrap (400× per fold): C3 − baseline mean Δ = **+0.006**, sd 0.014,
P(Δ>0) = 0.69. Fold 2 stays negative. C3 val-fold permutation drop:
`i_dso_x_dsr` 0.022, `i_transfer_x_ss` 0.009, `i_io_x_dsr` 0.003 — used,
but they steal from `f_ds_r` rather than add 0.02. Best substitution
bootstrap Δ = +0.012, P(Δ>0) = 0.62. swap+SHAP bootstrap Δ = +0.010,
P(Δ>0) = 0.55 (fold 0 +0.061, fold 1 −0.013). Neither is a KEEP. Learning curve:
lift is +0.004 at 25–75% of train companies, +0.009 only at 100%.

## What failed

- SHAP-pair add (the planned A) is flat.
- Dumping all legal I (B) is flat and the miss flags / zeroin / gap hurt.
  Twelve random legal triples all sit at 0.746–0.755; C3 (0.7615) is the
  best triple and still misses 0.772.
- Lags of I do not help (acf1 ≈ 0 on the transfer products).
- 400+ES collapses (1–6 trees) — same failure as `gbm_core`.
- No hyperparameter on the 50-tree family (trees 25–200, depth 1–5,
  colsample, lr) puts an I add-on at 0.772.
- Observed-row CV (15-col ± that column): `i_dso` 0.789→0.788,
  `i_gap` 0.792→0.796, `i_io_x_dsr` 0.750→0.749, SHAP pair +0.001.
  No legal i_* lifts where it is actually defined.

## Next idea

Do not merge I. Dropping `f_ds_r` with **no** `i_*` already lifts fold 0
(0.689→0.726) and holds mean CV at 0.7525 (bootstrap Δ 0.000,
P>0=0.37 — same fold-0 / fold-1 trade). C3 on top of that drop is only
+0.0046 (200×/fold, P>0=0.68) and **hurts fold 0 by −0.019**.
`i_dso_x_dsr` does not lift on the rows where it is defined (0.788 vs
0.789). If someone wants a stabler 15-col quote, use **shallower trees**
(depth 1–2 baseline 0.758) or drop `f_ds_r` from the core — that is a
core edit, not a Family I merge. Do not grow B. Do not revive 400+ES.
