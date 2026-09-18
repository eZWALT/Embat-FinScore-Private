# 2026-09-18-2135 — overnight metrics

- **Author:** agent
- **When:** 2026-09-18

## Decision

No official label overnight. Rank health-score candidates on **holdout proxies** in `overnight/`:

1. Maximize `(auroc_stress_lead3 + auroc_recover_lead3) / 2`
2. Then `lead_months`
3. Gates: coverage, 0–100, not a size proxy, no holdout leak, explainable scorecard

Labels are stress/recover from bank net flow + overdue/pending invoices — **not** from the score. Holdout: 72 companies / 15 groups, seed `20260918`, file `overnight/splits/holdout_companies.csv` (do not edit).

Do not maximize PCA variance, PLS, or in-sample fit.
