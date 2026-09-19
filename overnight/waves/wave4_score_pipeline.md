# Wave 4 — score_pipeline 14-signal reconcile (end note)

Agent `851eac72`. Files: `analysis/evaluate/score_pipeline_qa.py`, `analysis/outputs/score_pipeline_qa.md`, `score_pipeline_rho.png`, registry Spearman/coverage. No 0–100. No pillars. No product/. Did not call `run_score` / `fit_ref`. Did not edit `score_pipeline.py` or Family B.

## Return

- **11 SAME / 2 CLOSE / 1 DRIFT** on primary twins.
- Worst drift: `volatility` vs `b_bal_vol` ρ=0.354 n=14968 (per-company p50=0.232; 88% of books DRIFT). Reconstruct from A is identity — no named `a_vol`.
- CAT_MAP parity: IDENTICAL (22/22).
- The 14 do not include days-with-tx / issued_lag1.
- Concentration is top1 (ρ=0.995), not HHI (ρ=0.990; store top1↔HHI 0.994). HHI PARK as gradient X.
- CLOSE: `ar_overdue` 0.918 / `ap_overdue` 0.883 vs all-open store. 3m-window reconstruction is identity. ~71% of last-month open |amount| is older than 3m.
- Invoice-dark train companies: 470 (`COMP_0962` refund ghost). Schedule as-of rate 38 (`COMP_1027` after-snapshot). Fold ρ stable except vol (always DRIFT). None of the 14 is SIZE.

## What failed

Nothing to fix in owned files. `b_bal_vol` is the wrong object for Javier vol; Family D keeps PDI invoices that Javier drops (conc residual grows toward extract, rank stays SAME). Do not invent `a_vol` tonight (cashflow.py not owned).

## Next idea

If someone later names cashflow vol, it is `min(3, sd6(a_net)/max(mean6(a_op_in),1))`. Do not average the 14 (B↔invoice median |ρ|=0.038). Do not percentile-fit (frozen score).
