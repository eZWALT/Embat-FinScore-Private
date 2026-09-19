# Wave 1 slot 8 — Y5 payment behaviour + evaluate protocol

- **When:** 2026-09-18 ~23:56–00:10 CEST
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Holdout:** `analysis/splits/holdout_companies.csv` (72 companies / 15 groups, seed 20260918). Never used to fit a pooled percentile.

## Files written

- `analysis/targets/y5_payment.py`
- `analysis/evaluate/protocol.py`
- `overnight/waves/wave1_slot8.md` (this file)

No other files touched. No models fitted.

## Y5 columns

`META.forbidden_x_families = ["e"]`. Horizon = 3 future **calendar** months. Invoice construction follows `score_pipeline._invoice_features` (3-month issuance window for open stock, 3-month payment window for delay, delay clipped [-30, 120], early months masked for left truncation). Overdue is Banque de France **>30 days** (`due < period_end - 30d`), not any-overdue.

| column | definition |
|--------|------------|
| `y5_ap_od30_ownp80` | AP od30 share **> own-history p80** (months ≤ t only, ≥6 finite past months) in each of t+1, t+2, t+3 |
| `y5_ar_od30_sust` | AR od30 share **> own-history p80** (same rule). Train-pooled expanding p80 was tried first |
| `y5_ap_delay_up15` | amount-weighted AP delay **> trailing-6 mean + 15 days** (baseline at t) in each of t+1, t+2, t+3 |

p80 is never computed on holdout companies as a pool, and never uses months > t.

## Train acceptance (1214 companies, 21,157 company-months)

Base rate among **labeled** train company-months. Size score = `log1p(|op_in|)` (missing inflow filled with 0). Two-sided size AUROC = `max(auc, 1-auc)`; reject if ≥ 0.60.

| Y | labeled | coverage | n_pos | companies | base rate | size AUROC | two-sided | verdict |
|---|---------|----------|-------|-----------|-----------|------------|-----------|---------|
| `y5_ap_od30_ownp80` | 4,905 | 23.2% | 418 | 524 | **8.52%** | 0.543 | 0.543 | **ACCEPTED** |
| `y5_ar_od30_sust` | 3,315 | 15.7% | 237 | 393 | **7.15%** | 0.519 | 0.519 | **ACCEPTED** |
| `y5_ap_delay_up15` | 3,408 | 16.1% | 153 | 442 | **4.49%** | 0.469 | 0.531 | **REJECTED** (rate < 5%) |

Labeled window: od30 Ys 2025-04 .. 2026-05; delay Y 2025-08 .. 2026-05 (needs 6 delay months after the pipeline mask). Events barely overlap (34 AP∩AR, 8 AP∩delay, 5 AR∩delay positives) — not the same column.

## What failed

- **Train-pooled AR p80** (expanding, holdout out, periods ≤ t): calendar-correct 3-month persistence = **2.84%**. Row-shifting an invoice-only panel had inflated this to ~6% by skipping months with no open AR. Own-history p80 is the allowed fallback and clears 5–30%.
- **`y5_ap_delay_up15`** at the specified +15 / trailing-6 / all-3-calendar-months is **4.49%** — just under the floor. Size is fine.

## Protocol functions (`analysis.evaluate.protocol`)

Required:

- `load_holdout()` → 72 ids
- `train_companies(con)` → 1,214 companies with `group_id`
- `group_folds(companies_df, n=5, seed=20260918)` → whole `group_id` in one fold (5×47 train groups)
- `assert_no_holdout(index_or_ids)`
- `leakage_check(X_cols, y_col, forbidden_prefixes)`
- `auroc`, `pr_auc` (numpy/pandas; no sklearn)

Also: `HOLDOUT_PATH`, `FOLD_SEED`, `rolling_origins(periods, horizon=3, n_last=9)`.

Smoke: `assert_no_holdout(holdout)` raises; `leakage_check(["e_ar_overdue"], "y5_ar_od30_sust", ["e"])` fails; `auroc`/`pr_auc` = 1.0 on a perfect ranking.

## Next idea

- To rescue delay without dropping the 15-day spec: trailing-6 with `min_periods=4` (invoice gaps) → 5.22% ACCEPTED; or 3-month **mean** delay > baseline+15 (plan’s “3-month aggregate”) → 10.4%. Do not relax to +10 days unless the 15-day Banque/Hirshleifer reading is abandoned.
- Train AR p70 (not p80) would also accept at 10.1% if a **level** “high AR” (vs own-history deterioration) is wanted later.
- Family E must stay out of X for these Ys. Cross-source test is cash/ops/debt at t → Y5 at t+3.
