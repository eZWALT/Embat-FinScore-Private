# Wave 4 — delay / overdue leftover

Agent `47815ca4`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/delay_qa.py`
- `analysis/outputs/delay_qa.md`
- `analysis/outputs/delay_vs_dso.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `invoices.py`, `gbm_y7_core.py`, `credit_note_qa.py`, `zero_in_qa.py`, `growth_qa.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. Days 0.711 untouched.

## Locked verdict

| object | decision |
| --- | --- |
| delay_coll as Y7 leftover | **KEEP** |
| delay / overdue as Y7 add-on | **CLOSE** (do not grow TURNOVER) |
| delay_paid as Y7 leftover | **CLOSE** |
| ar_overdue as Y7 leftover | **CLOSE** |
| delay / overdue as Y3 X / the 44 | **CLOSE / DROP from the 44** |
| delay as a health Y | **PARK** |
| Q6 delay on short books | **CLOSE** |
| 3m-vs-all-open leftover | **CLOSE** |

ρ delay vs DSO 0.214. Y7 leftover after DSO 0.581 / issued_lag1 0.583 / both 0.584. Same-n leftover 0.584 ≈ raw 0.579 ≈ issued 0.578. ICC 0.922 demean 0.521. Y3 delay 0.512 leftover-after-days 0.427 vs days 0.711. 3m leftover 0.509. Q6 short lag1 0.576 early cov 0.0% after-month-7 lag1 0.570 vs issued 0.643. Fold 4 delay 0.576 vs DSO 0.342 vs issued 0.647.

## What failed / next

- Y3 leftover after days 0.427 — CLOSE as Y3 X
- CLOSE as Q6 — delay empty until month 7
- 3m-vs-all-open leftover dies — CLOSE window is not a health lever
- fold 4 delay 0.576 does not save the short-DSO hole
- same-n leftover 0.584 ≈ raw 0.579 — not a new object
- BETWEEN style: company-mean 0.569 demean 0.521 ICC=0.922
- delay_paid leftover dies 0.449 — CLOSE
- AR overdue leftover after DSO dies 0.547 — CLOSE
- same-n leftover − issued Δ=0.005 — do not grow TURNOVER

Elapsed 9s.
