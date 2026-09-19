# Wave 4 — a_in3 leftover after days (size control)

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/in3_qa.py`
- `analysis/outputs/in3_qa.md`
- `analysis/outputs/in3_leftover.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `salary_month_qa.*`, `ss_qa.*`, `issued_qa.*`, `cashflow.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.

## Locked verdict

| object | decision |
| --- | --- |
| `log1p(a_in3)` as 44-col size control | **CLOSE as unused leftover** |
| `a_in3` as engine X / 15-col card | **no** |

Y3 leftover after days rank 0.521 (dies=True, fake=False); inverse days after size 0.667. Single size 0.617 vs days 0.711. ρ vs days 0.595. Q6 lag1 leftover 0.527. unused leftover after days: honest rank 0.521 dies (OLS 0.531 fake=False). DROP from the 44 as engine X. Size bar 0.617 stays the KEEP-as-X quote, not a card stem.

## What failed / next



Elapsed 39s.
