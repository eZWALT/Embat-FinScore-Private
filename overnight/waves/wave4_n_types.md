# Wave 4 — f_n_types leftover after days

Agent `b17e9c44`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/n_types_qa.py`
- `analysis/outputs/n_types_qa.md`
- `analysis/outputs/n_types_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not overwrite `factoring_qa.*`, `ds_r_qa.*`, `fc_r_qa.*`, `tax_month_qa.*`, `n_cust_qa.*`, `issued_qa.*`, `in3_qa.*`. Did not touch `debt.py`, `gbm_core.py`, the 15-col card, TURNOVER, product/, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. TURNOVER **0.720 / 0.712**.

## Locked verdict

| object | decision |
| --- | --- |
| leftover after days | **CLOSE** 0.534 |
| as Y3 X | **DROP from the 44 as Y3 X** |
| leftover after f_n_facilities | **0.515** rewrite=False |
| days leftover after n_types | **0.688** |
| twin vs facilities | **True** ρ=0.994 |
| rise-only | **True** rises=344 drops=0 |
| Q6 | **CLOSE** |

Y3 n_types 0.578 vs days 0.711 vs size 0.617 vs facilities 0.579. Leftover after days rank 0.534 OLS 0.530 ρ(resid,days)=-0.360. lag1 leftover after days_lag1 0.529. Bootstrap leftover-after-days p05/p50/p95 0.425 / 0.533 / 0.573. After facilities boot p50 0.518. Ever-connected recover 13.3% vs never 29.6%. Rise leftover after days is a fake days leak ρ=-0.943. Mixed-rise leftover after days 0.471. Δ leftover after days 0.607 KEEP_Δ=False. g_n_types leftover 0.570. Last Y3-month leftover 0.471. First Y3-month leftover 0.482. Residual ICC 0.984.

## What failed / next

- no replica miss; leftover after days CLOSE 0.534 (dies) and twin of f_n_facilities ρ 0.994 → DROP from the 44. T1 leftover sometimes lives (boot p50 0.579) but raw loses to size and the twin gate still kills KEEP.

Elapsed 39s.
