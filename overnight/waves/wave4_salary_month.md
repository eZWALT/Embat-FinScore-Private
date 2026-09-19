# Wave 4 — c_salary_month leftover after days

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/salary_month_qa.py`
- `analysis/outputs/salary_month_qa.md`
- `analysis/outputs/salary_month_leftover.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `salary_qa.py` / `.md`, `ss_qa.*`, `ds_r_qa.*`, `issued_qa.*`, `ops.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.

## Locked verdict

| object | decision |
| --- | --- |
| `c_salary_month` on the 15-col card | **KEEP** |
| calendar | **monthly** |
| `c_missed_salary` as Y3 X | **CLOSE** (already) |

Y3 leftover after days rank 0.603 (dies=False, fake=False); inverse days after salary 0.647. After SS 0.626. Single 0.671 vs days 0.711 vs size 0.617. ρ vs days 0.419. Q6 lag1 leftover 0.606. leftover after days rank 0.603 lives, beats size by 0.055, not SIZE, not a twin

## What failed / next



Elapsed 45s.
