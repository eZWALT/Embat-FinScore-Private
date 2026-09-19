# Wave 2 · slot 4 — baselines (Y1 naive, Y2 single-feature, Javier)

- **Files:** `analysis/models/baselines.py` (owned). Registry rows appended only (header untouched). This note.
- **Not edited:** family F / `debt.py`, `product/`, other agents' files.
- **Re-run:** `python -m analysis.models.baselines` (idempotent registry keys: agent, y, model, split, metric).
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Holdout:** 72 companies / 1,073 company-months. Metrics on holdout only. Naive methods have no fit. Javier `fit_ref` on 21,157 train company-months (`assert_no_holdout`). Feature signs chosen on train.
- **Y2 X:** families A, C, D, E, F, G, H (92 numeric cols). **Never B.** `leakage_check` ok. Nothing skipped.

## Y1 naive (holdout) — last_value / hist_mean / seasonal_naive_12

Forecast at origin t uses only the cash panel at or before t. Seasonal is Hyndman m=12: ŷ_{t+h} = y_{t+h-12}. MASE scale = train-only mean |y_t − y_{t-12}| (net 5.16e6, op_in 6.41e6, liq 6.98e6).

Holdout n_y: h1 = 1,001; h3 = 857. last_value and hist_mean coverage = 1.0. seasonal coverage = 0.284 (h1) / 0.331 (h3) — only 284 company-months have a 12-month lag.

### All holdout Y (primary)

| y | last MASE | last sMAPE | mean MASE | mean sMAPE | seas MASE | seas sMAPE | seas cov |
|---|----------:|-----------:|----------:|-----------:|----------:|-----------:|---------:|
| y1_net_h1 | 0.408 | 1.254 | **0.341** | 1.332 | 0.093 | 1.391 | 0.284 |
| y1_net_h3 | 0.365 | 1.265 | **0.361** | 1.362 | 0.093 | 1.391 | 0.331 |
| y1_in_h1 | 0.570 | **0.790** | **0.452** | 0.893 | 0.151 | 0.953 | 0.284 |
| y1_in_h3 | 0.512 | **0.801** | **0.496** | 0.921 | 0.151 | 0.953 | 0.331 |
| y1_liq_h1 | 1.794 | **0.486** | 2.094 | 0.617 | 0.095 | 0.871 | 0.284 |
| y1_liq_h3 | 1.901 | **0.634** | 2.467 | 0.706 | 0.095 | 0.871 | 0.331 |

Mean across series: last_value MASE 0.924 / 0.926 (h1/h3), sMAPE 0.843 / 0.900. hist_mean MASE 0.962 / 1.108, sMAPE 0.947 / 0.996. seasonal MASE 0.113 looks strong only because coverage is thin and the train scale is dominated by large/early months.

### Same 284 rows (seasonal support) — seasonal loses

| series | h | last MAE | mean MAE | seas MAE | last sMAPE | mean sMAPE | seas sMAPE |
|--------|--:|---------:|---------:|---------:|-----------:|-----------:|-----------:|
| net | 1 | 3.97e5 | **3.31e5** | 4.81e5 | **1.212** | 1.337 | 1.391 |
| net | 3 | 3.88e5 | **3.41e5** | 4.81e5 | **1.221** | 1.353 | 1.391 |
| op_in | 1 | 1.32e6 | **8.53e5** | 9.68e5 | **0.683** | 0.874 | 0.953 |
| op_in | 3 | 8.30e5 | **8.14e5** | 9.68e5 | **0.660** | 0.889 | 0.953 |
| liq | 1 | **2.05e5** | 4.50e5 | 6.61e5 | **0.493** | 0.760 | 0.871 |
| liq | 3 | **3.02e5** | 4.96e5 | 6.61e5 | **0.635** | 0.793 | 0.871 |

**Headline:** hist_mean wins Y1 flows (MASE). last_value wins Y1 liquidity (sMAPE). seasonal_naive_12 is not the number to beat.

## Y2 single-feature AUROC vs `y2_neg_2of3`

Holdout labels: **857 rows, 23 positives, base rate 2.68%** (train was 7.32% / 1,271 pos). AUROCs on 23 events are noisy. Sign of each feature chosen on train.

### Top 5 holdout AUROCs (requested)

| feature | fam | sign | train AUROC | **holdout AUROC** | cov |
|---------|-----|------|------------:|------------------:|----:|
| h_n_siblings_active | H | − | 0.503 | **0.856** | 1.00 |
| f_fc_r | F | + | 0.521 | **0.700** | 0.83 |
| e_delay_coll | E | − | 0.565 | **0.692** | 0.39 |
| e_ar_overdue | E | − | 0.570 | **0.690** | 0.43 |
| h_share_group_in | H | + | 0.519 | **0.675** | 0.98 |

`h_n_siblings_active` is a holdout fluke: train means y0/y1 are 9.06 vs 9.11 (AUROC 0.50); holdout stressed months sit in small groups (1.96 vs 9.26 siblings). Do not treat 0.856 as a real signal.

### Top 5 by train AUROC (honest pick → holdout)

| feature | train AUROC | holdout AUROC |
|---------|------------:|--------------:|
| c_n_tx / a_n_tx | 0.601 | 0.415 |
| c_gap_sd | 0.585 | 0.345 |
| a_in12 | 0.579 | 0.215 |
| c_n_days_with_tx | 0.577 | 0.382 |

Size/activity features that clear ~0.58–0.60 on train reverse on holdout. `e_ar_overdue` (train 0.570 / holdout 0.690) is the least-flaky allowed single feature.

## Javier score

`build_features` → `fit_ref` on train company-months only → `run_score` with that ref → AUROC of **−score** vs `y2_neg_2of3` on holdout.

- **AUROC = 0.776**
- Scored 714 / 857 holdout labels (83.3%; `cov_w >= 0.5`). Among scored rows coverage was 1.0.
- Uses the 14 pipeline signals including runway / d_runway / neg_liq (**family B**). Not a fair competitor for no-B models. It is the existing score’s persistence vs the same liquidity path that defines Y2.

## What failed

- Seasonal MASE on unmatched support is misleading (low coverage). On matched rows it loses to last_value and hist_mean.
- Holdout Y2 has only 23 events; ranking features by holdout AUROC overfits the 72-company split.
- Train-best single features are size proxies and do not generalise.
- Javier 0.776 is not an allowed-X baseline.

## Next idea

- Monthly GBM / TS must beat **hist_mean** on net/in and **last_value** on liq (sMAPE), not seasonal MASE 0.11.
- Fair Y2 baseline for no-B models is ~0.57 train / ~0.69 holdout (`e_ar_overdue`), not Javier 0.78 and not `h_n_siblings_active` 0.86.
- Report bootstrap CI over companies; 23 holdout positives will not decide a bake-off.
