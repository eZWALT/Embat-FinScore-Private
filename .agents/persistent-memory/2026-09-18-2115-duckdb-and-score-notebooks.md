# 2026-09-18-2115 — DuckDB layer + EDA and score notebooks

- **Author:** Javier Boix (with Claude Code)
- **When:** 2026-09-18

## What changed

- `analysis/build_db.py` builds `data/embat.duckdb` (gitignored) from the CSVs in `data/` or `data/raw/output/`; `analysis/requirements.txt` has `duckdb`.
- `analysis/eda.Rmd`: EDA of the 8 tables, reads from DuckDB.
- `analysis/02_score_salud_financiera.Rmd`: monthly signals -> 5-pillar score (frozen percentile reference) -> trajectory/states -> bache vs fall -> explanation -> proxy-event backtest -> monitor -> dynamic credit line + recommendation agent. Exports `analysis/outputs/*.csv`.

## Decisions

- No labels exist, so the score is interpretable/unsupervised, not a trained model. Validation uses proxy events on future cash and group-level folds.
- Invoices cover only 785/1286 companies; `payment_date` is dirty (years 2000-6913) and a placeholder for unpaid invoices. Only trusted when `status = 'paid'` and inside the window.
- Assumption to verify: invoice `amount > 0` = issued (AR), `< 0` = received (AP).
- Multi-currency amounts are used unconverted.

## Still unknown

- Notebooks were **not executed** (no R where they were written). First knit may need fixes.
- Pillar weights and the 0.75/0.25 level/trend mix are priors, not tuned. Buyer/product/demo not final.
