# Wave 4 — f_util_snapshot leftover after days

Agent `b17e9c44`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/util_snap_qa.py`
- `analysis/outputs/util_snap_qa.md`
- `analysis/outputs/util_snap_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not overwrite `debt_schedule_qa`, `ogtg_qa.*`, `factoring_qa.*`, `n_types_qa.*`, `ar_open_qa.*`, `ap_open_qa.*`. Did not touch `debt.py`, `gbm_core.py`, the 15-col card, TURNOVER, product/, FROZEN_ACCEPTED, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. TURNOVER **0.720 / 0.712**.

## Locked verdict

| object | decision |
| --- | --- |
| leftover after days | **CLOSE** hole 0.711 native n_pos=0 |
| as Y3 X | **PARK as snapshot / DROP from the 44 as Y3 X** |
| last-month-only | **True** cov 1.6% vs OGTG 5.7% |
| leftover after n_types | **0.578** |
| leftover after OGTG | **0.500** |
| days leftover after hole | **0.711** |
| extract-hole / Y10 impossible | **True** / **True** |
| Q6 | **CLOSE** |

Y3 native — hole-fillna0 0.500 dummy 0.500 vs days 0.711 vs size 0.617 vs n_types 0.578. ρ days 0.034 types -0.041 OGTG 0.152. lag1 leftover after days_lag1 0.632. Trait leftover after days 0.622 fake=False. Bootstrap trait p05/p50/p95 0.391 / 0.607 / 0.693. Defined-dummy leftover 0.711 fake=True. Trait leftover after last-month days+n_types 0.617 boot p50 0.572. Logo min 0.544. Permute p(obs≥null)=0.042. Last-month dummy leftover after days+n_types 0.530 dies. Positive-util dummy leftover 0.473 dies. Inverse OGTG after util 0.642. Q6 inverse days after util0_lag1 0.711. Penta boot p05/p50/p95 0.375 / 0.605 / 0.697. Logo-penta min 0.579. Hexa leftover 0.626. Permute-penta p(obs≥null)=0.000. OGTG PARK leftover 0.540. Inverse days after util+OGTG+n_types 0.652.

## What failed / next

- no replica miss; native leftover undefined (Y3 n_pos=0 on 2026-08). fillna0 / defined-dummy leftover 0.711 is a fake days leak (ρ=-0.959). Trait leftover 0.622 is not a days leak (ρ=0.044) but folds 0.424–0.760 / boot p05 0.391 dies. PARK snapshot X. Utilisation impossible as a Y.

Elapsed 52s.
