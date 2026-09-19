# Wave 4 — shrink global Y3 to a core (`gbm_core`)

Agent `7f37f0fd`. Two global LightGBMs on `y3_recover_cash_6m` (45→65 / who is improving) plus a depth-3 50-tree diagnostic of A. No product. No 0–100. Holdout never in a fit. Quote **train group-fold CV** only.

Re-run:

```bash
/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.gbm_core
```

## Files written

- `analysis/models/gbm_core.py` (owned)
- this note
- 8 rows appended to `analysis/experiments/registry.csv` (A / B / shallow-A AUROC + sd; two train-fold singles)

Did not edit `analysis/models/gbm_y3y6.py`. Per-group Y3 stays PARKED (CV 0.694 < 0.710). Family B never in X. `a_op_in` dropped (SIZE). No parquet / duckdb write.

## Setup

| item | choice |
|------|--------|
| Y | `y3_recover_cash_6m`, stressed/labeled rows only (NaN otherwise) |
| Train | 5648 / 402 / 7.12% (5648/21157 = 26.7% of train company-months) |
| Split | 5 `group_folds`, seed `20260918`. Holdout 72 companies excluded from every fit and every val fold. Not scored (LOW_POWER). |
| Lags | 1 and 3 of the listed stems only. No `group_size` / `n_banking`. |
| A / B trees | same as published 0.710 (`LGB_BASE` + early stopping 40) |
| Shallow-A | night 0.762 spec: 50 trees, `max_depth=3`, `num_leaves=8`, no early stop |
| DSO clip | fixed 24-month upper clip on `e_dso_proxy` / `e_dpo_proxy` (constant, not a fit) |

**A stems** (SHAP + perm core; optional `e_dso_proxy` left out so A stays smaller): `c_ss_month`, `c_salary_month`, `a_n_tx`, `f_ds_r`, `c_n_days_with_tx`. **n_x=15**.

**B stems** (feature_report recommended-keep ∩ ratios/flags; never B; no raw euro; no counts): 42 stems including payroll/tax month flags and `c_gap_sd` instead of `a_n_tx`. **n_x=126**.

## Quote (train CV only)

| spec | model | mean AUROC | fold sd | n_x | vs 0.710 | vs dummy 0.50 | vs best single |
|------|-------|----------:|--------:|----:|----------|---------------|----------------|
| **A** | `lgbm_y3_core_shap` | **0.7099** | 0.041 | **15** | −0.0001 | yes | no (`c_n_days_with_tx` 0.711) |
| **B** | `lgbm_y3_core_keep` | **0.6929** | 0.017 | **126** | −0.017 | yes | yes (`c_gap_sd` 0.671) |
| **A shallow** | `lgbm_y3_core_shap_d3_n50` | **0.7520** | 0.037 | **15** | +0.042 | yes | yes (0.752 > 0.711) |
| published global | `lightgbm_y3y6` | 0.710 | — | 278 | — | — | — |
| published shallow | `lgbm_small_d3_n50` | 0.762 | 0.016 | 278 | — | — | — |

Folds A (early-stop trees): 0.662 (1), 0.761 (7), 0.707 (1), 0.681 (1), 0.739 (3).
Folds B (early-stop trees): 0.676 (1), 0.698 (10), 0.680 (1), 0.719 (2), 0.692 (3).
Folds shallow-A (50 / depth 3): 0.689, 0.780, 0.769, 0.752, 0.769.

## KEEP / PARK

Rule: KEEP if A or B CV ≥ 0.710 with fewer columns (prefer smaller). PARK if both lose by > 0.02.

- A = 0.7099: miss by 0.0001 (rounds to **0.710**). Early stopping collapsed to 1–7 trees, so this number is almost the single `c_n_days_with_tx` (0.711).
- B = 0.6929: lose by 0.017 (inside the 0.02 band). Larger and worse than A.
- Shallow-A is diagnostic, not in the KEEP gate. It **holds**: 0.752 vs full-X shallow 0.762 (gap 0.010) and it does beat 0.710 / dummy / single.

**Decision: CLOSE — do not PARK. Do not KEEP A/B on the 0.710 early-stop spec.** Prefer A if a core must be named (15 cols, ties 0.710). The number that actually holds is **shallow-A 0.752 / n_x=15**.

## What this means (six questions)

Quiet-stressed recovery (Q2 who is improving; Q5 why) lives in five stems: no SS / salary booking, fewer txs and booking days, lower debt-service ratio. Dropping `a_op_in` (SIZE) does not kill the signal. The 278-col global engine is mostly unused capacity once trees are allowed to grow (depth-3, 50 trees). Family B is still out — this is not a leaked runway copy.

## What failed

- Early stopping on the 0.710 spec is unusable on the tiny X (1 tree on 3/5 A folds). Same failure mode as the Y2 panel note.
- B’s extra keep-list ratios/flags did not recover the 0.017. `c_gap_sd` is a weaker single (0.671) than `c_n_days_with_tx` (0.711); the cluster-rep swap costs the activity signal.
- A under early stop does **not** beat the best single.
- Optional 6th `e_dso_proxy` was not added (prefer smaller; SHAP/univ sign flip).

## Next idea

Promote the **15-col depth-3 / 50-tree** engine as the night Y3 quote (0.752 vs 0.762 full-X). If someone wants a KEEP on the 0.710 protocol, refit A with a 50-tree floor (no early stop) rather than adding columns. Do not grow B. Do not revive per-group Y3.
