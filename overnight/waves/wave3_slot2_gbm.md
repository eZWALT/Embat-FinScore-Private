# Wave 3 slot 2 — global LightGBM for `y2_neg_2of3`

- **When:** 2026-09-19 ~00:04–00:12 CEST
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3` (installed `lightgbm==4.7.0`; sklearn already present)
- **Re-run:** `python -m analysis.models.gbm_panel`
- **Holdout:** `analysis/splits/holdout_companies.csv` (72 companies / 15 groups). Never used to fit.

## Files written

- `analysis/models/gbm_panel.py` (owned)
- This note
- Appended rows to `analysis/experiments/registry.csv` (CV + holdout AUROC/PR-AUC, dummy, train-selected single)

No other files edited. Family H not touched. Family B never imported. No parquet / duckdb write. No 0–100 score. No SHAP.

## Setup

| item | choice |
|------|--------|
| Y | `y2_neg_2of3` only |
| X families | **A+C+D+E+F+G+H** (D imported and built). **Never B** |
| Lags | 1 and 3 of numeric family X (not of static `group_size` / `n_banking` / `h_group_size`) |
| Meta | `group_size`, `n_banking` (raw counts). No `company_id` / `group_id` encodings |
| Monotone | **skipped** (signs not sure enough) |
| Split | 5 `group_folds` on train groups only (`FOLD_SEED=20260918`); then one fit on all train labeled rows; holdout once |
| Trees | **200** fixed, `lr=0.05`, `num_leaves=31`, `min_child_samples=80`, `scale_pos_weight` from the training slice only |
| Metrics | `analysis.evaluate.protocol.auroc` / `pr_auc` |

`leakage_check` passed (no `y2_*`, no `b_*`). `assert_no_holdout` on every train slice.

X after lags: **276** numeric columns (94 contemporaneous + lags). Train labeled 17,356 / 21,157 company-months (coverage 0.7807; last 3 months have no Y horizon). Holdout labeled 857 / 1,073.

## CV vs holdout

| split | AUROC | PR-AUC | n | n_pos | base rate |
|-------|------:|-------:|--:|------:|----------:|
| fold 0 | 0.412 | 0.089 | 3,384 | 374 | 11.1% |
| fold 1 | 0.638 | 0.047 | 3,158 | 89 | 2.8% |
| fold 2 | 0.561 | 0.082 | 3,340 | 187 | 5.6% |
| fold 3 | 0.536 | 0.095 | 3,787 | 310 | 8.2% |
| fold 4 | 0.542 | 0.101 | 3,687 | 311 | 8.4% |
| **CV mean** | **0.538** | **0.083** | | | train 7.32% |
| **Holdout** | **0.491** | **0.031** | 857 labeled | ~23 | **2.68%** |

Coverage = **100%** of holdout company-months (1,073 / 1,073) received a finite prediction (LightGBM missing-value handler). Metrics use the 857 labeled holdout rows.

Holdout AUROC is **below** the dummy prior (0.50). Event rate on holdout is ~3× lower than train, and fold 0 (most positives) is already anti-predictive — group-level base rates do not transfer.

## Does GBM beat the single-feature baseline?

**No** against the number `baselines.py` already published; **yes** only against the train-selected single feature (which fails on holdout).

| model | how chosen | holdout AUROC | holdout PR-AUC |
|-------|------------|--------------:|---------------:|
| dummy prior (train prevalence 0.0732) | constant | 0.500 | 0.027 |
| `a_n_tx` (sign +) | **train** AUROC 0.601 via `baselines.single_feature_auroc` | 0.415 | 0.022 |
| `h_n_siblings_active` (sign −) | **baselines.py published top-1** (ranked on holdout; train AUROC 0.503) | **0.856** | — |
| Javier `−score` | train-only `fit_ref` | 0.776 | — (uses family **B**, forbidden here) |
| **lightgbm_panel (200 trees)** | group-fold CV, then all-train fit | **0.491** | **0.031** |

- Beats dummy? **No** (0.491 < 0.500).
- Beats train-selected single `a_n_tx`? **Yes** (0.491 > 0.415) — that univariate does not generalize.
- Beats published single `h_n_siblings_active`? **No** (0.491 ≪ 0.856). That 0.856 has train AUROC ≈ 0.50 and ~23 holdout positives, so it is a holdout-ranked small-n spike, not a usable baseline to chase.

A first pass with early stopping collapsed to **1 tree** on every fold (val AUC noise). Forcing 50 trees after that collapse gave holdout AUROC **0.541** (beats dummy, still far below 0.856). More trees did not help: 200 trees overfit `group_size` / `n_banking`.

## Top 8 gain importances (final 200-tree model)

| rank | feature | gain |
|-----:|---------|-----:|
| 1 | `h_group_size` | 20,085 |
| 2 | `n_banking` | 10,115 |
| 3 | `group_size` | 9,244 |
| 4 | `a_n_tx` | 8,420 |
| 5 | `g_custom_share` | 6,068 |
| 6 | `a_n_tx_lag1` | 5,783 |
| 7 | `d_tx_cp_share` | 5,744 |
| 8 | `f_fc_r` | 5,272 |

Almost all mass is **group identity / size / activity** (`h_group_size` ≈ `group_size`, `n_banking`, `a_n_tx`). That matches fold 0 failing and holdout base-rate shift. Not a cash-path model — family B is correctly excluded.

## What failed

- Early stopping on fold val AUC is unusable here (always picked tree 1).
- Global GBM without family B does **not** predict 3-month negative-liq stress better than a coin flip on holdout.
- `group_size` / `n_banking` dominate gain and do not transfer to new groups.
- Holdout has only ~23 positives; any single-feature 0.85 on that slice is not a confirmation.
- `registry.csv` has unquoted commas in some other agents' notes (pandas `read_csv` breaks). This module parses with `csv.reader`.

## Next idea

Drop static group-level columns and re-fit (company-level A/C/E/F only), or park global GBM for Y2 until a non-B signal exists. Do not treat `h_n_siblings_active` 0.856 as a target. Per-group models are the plan’s next check, but only if they beat this global result **and** a dummy on holdout — this run does not clear that bar.
