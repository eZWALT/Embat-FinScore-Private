# Wave 3 · slot 7 — LightGBM Y3 cash-recovery + Y6 missed-payroll

- **Files:** `analysis/models/gbm_y3y6.py` (this note). Did **not** edit `y3_recovery.py`, `y6_activity.py`, or `gbm_panel.py`.
- **X store:** `data/feature_store/monthly.parquet` 22,230 × 118 (A–H).
- **Y:** existing `build` on `y3_recover_cash_6m` and `y6_missed_payroll` (accepted wave-2 binaries).
- **Protocol:** 5 group-fold CV (`FOLD_SEED=20260918`) on TRAIN groups only; one fit on all train; one holdout pass. Dummy = constant train prevalence (AUROC 0.5). Best single = train-best allowed contemporaneous feature with ≥25% labeled-train coverage (min 200 rows); sign from train.
- **Lags:** 1 and 3 of family columns. Meta kept: `group_size`, `n_banking` (not lagged). No monotone constraints.
- **Re-run:** `python -m analysis.models.gbm_y3y6`
- **Registry:** appended R3 / wave 3 / agent `d2753bb5` (use the **00:10** rows; an earlier 00:09 pass picked sparse `f_sched_vs_obs` / `f_w_rate` with undefined holdout AUROC).

## y3_recover_cash_6m — beats dummy **and** beats single

Stressed-only (`liq<0` or `runway<1` at t). Label already NaN off that set. Every labeled row matches `b_liq<0 | b_runway<1` (5,883 / 5,883). **X: A+C+D+E+F+G+H. Never B.**

| split | n labeled | pos | rate | LightGBM AUROC | dummy | best single | beats single |
|-------|----------:|----:|-----:|---------------:|------:|-------------|--------------|
| train / CV-5 group | 5,648 | 402 | **7.12%** | **0.710** | 0.50 | `c_n_days_with_tx` (−) train 0.723 | (CV ≈ univariate) |
| holdout | 235 | 14 | 5.96% | **0.899** | 0.50 | same feature holdout **0.797** | **yes** (0.899 > 0.797) |

- Train coverage of the stressed-labeled set: 5,648 / 21,157 = **26.7%** of train company-months (725 companies). Holdout prediction coverage on labeled rows = 1.0 (278 X cols after lags).
- Holdout has only **14** positives — 0.899 is noisy. The stable number is **CV 0.710**.
- Next singles (train AUROC): `a_n_tx` 0.714, `c_n_tx` 0.714, `c_gap_sd` 0.693, `c_ss_month` 0.690. Lower activity among the stressed predicts cash recovery (mean-reversion / quiet firms).
- Top GBM gain: `c_ss_month`, `c_salary_month`, `h_group_size_lag3`, `c_gap_sd_lag3`, `f_ds_r_lag1`.

**Verdict:** LightGBM beats dummy. On the official holdout comparison it also **beats the best allowed single feature**. Treat the 0.90 as a small-n bonus; CV says the model is about even with ops activity, a bit above 0.70.

## y6_missed_payroll — does **not** beat dummy or single

Usual-payroll rows only. **X: B+D+E+F+G+H. Never A or C.**

| split | n labeled | pos | rate | LightGBM AUROC | dummy | best single | beats single |
|-------|----------:|----:|-----:|---------------:|------:|-------------|--------------|
| train / CV-5 group | 6,004 | 367 | **6.11%** | **0.577** | 0.50 | `e_dpo_proxy` (+) train 0.613 | **no** |
| holdout | 271 | 7 | 2.58% | **0.483** | 0.50 | same feature holdout **0.685** | **no** (0.483 < 0.685) |

- Train coverage: 6,004 / 21,157 = **28.4%** of train company-months (579 companies). 212 X cols after lags. Holdout coverage on labeled = 1.0.
- Early stopping hit 1 tree on 4/5 folds — no usable multivariate signal once A/C are stripped.
- Holdout has only **7** events; 0.483 vs 0.500 is noise. CV 0.577 is the honest “weak” result.
- Next singles (train): `b_bal_vol` 0.599, `e_ap_overdue_30` 0.592, `e_ar_overdue_30` 0.586.

**Verdict:** LightGBM does **not** beat the dummy and does **not** beat the best single allowed feature. Park this Y for models that cannot use A/C.

## y6_zero_in_3 — CAUTION diagnostic only (not a clean win)

Same X as payroll (B+D+E+F+G+H). One-sided size AUROC 0.12 / two-sided 0.88 (wave 2). Do not treat as a bake-off win.

| split | n labeled | LightGBM AUROC | best single `h_share_group_in` (−) |
|-------|----------:|---------------:|-----------------------------------:|
| CV-5 | 17,083 | 0.924 | train 0.859 |
| holdout | 848 (65 pos) | 0.947 | holdout 0.904 |

GBM “beats” the single feature because both are reading the same inverse-size / solo-share path (`h_share_group_in` dominates gain). Not promoted.

## What failed

- **y6_missed_payroll + allowed X:** salary regularity lives in family C (and levels in A). Forbidding A/C is correct for leakage, and it leaves almost nothing to predict a 3-month payroll miss. Invoice DPO is a weak univariate (train 0.61) that the GBM does not improve on.
- **Holdout event counts** for both accepted Ys are tiny (14 and 7). Do not rank models on holdout AUROC alone.
- **Sparse F snapshot columns** (`f_w_rate`, `f_sched_vs_obs`, ~1.6% coverage) can fake 0.8+ train AUROC. Coverage gate is required for the single-feature baseline.

## Next idea (later wave; do not retune here)

Keep `y3_recover_cash_6m` as the recovery binary (X may include E; never B). A simpler activity-only model (`c_n_days_with_tx` / `a_n_tx`) is the baseline to beat on the next split — GBM’s CV lift over that univariate is small. For payroll, do not relax A/C at t; if the label stays, add a **new** lagged salary-regularity column from a window that does not overlap t+1..t+3 (not `c_salary_month` / `c_missed_salary` at t). Do not promote `y6_zero_in_3`.
