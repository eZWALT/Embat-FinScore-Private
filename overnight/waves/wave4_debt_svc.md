# Wave 4 — a_debt_service leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/debt_svc_qa.py`
- `analysis/outputs/debt_svc_qa.md`
- `analysis/outputs/debt_svc_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `ds_r_qa.*`, `fc_r_qa.*`, `op_out_qa.*`, `in3_qa.*`, `a_vol_qa.*`, `transfer_qa.*`, `growth_qa.*`, `n_accounts_qa.*`, `gbm_core.py`, `cashflow.py`, `debt.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged. `f_ds_r` leftover 0.528 DROP stays. `f_fc_r` leftover 0.449 DROP stays. KEEP `f_fc_r_lag3` on TURNOVER.

## Locked verdict

| object | decision |
| --- | --- |
| `a_debt_service` as Y3 X / 15-col card | **CLOSE unused leftover** |
| `a_debt_service` as engine X on the 44 | **DROP** |
| `y_debt_service` | **PARK** |
| TURNOVER | **CLOSE** — do not grow 0.720 |
| f_ds_r leftover | **DROP quote 0.528** |
| size bar | **KEEP quote 0.617** |

## Locked extras

- Honest leftover after days rank 0.484 (lives=False, OLS fake=False); inverse 0.673.
- Single 0.613 vs days 0.711 vs size 0.617 vs f_ds_r 0.620.
- SIZE=False twins=['f_ds_r', 'f_debt_service'].
- leftover after f_ds_r 0.587 rewrite=False. after days+size 0.487.
- Q6 lag1 leftover 0.522. Dark leftover 0.474 ERP 0.454.
- ICC=0.755 median company Pearson acf1=0.026 n_co=483. month shock / WITHIN.
- leftover after f_debt_service 0.613 a_fin_cost 0.559 a_op_out 0.519 a_n_tx 0.485 f_fc_r 0.595.
- demeaned a_debt_service Y3 0.449 leftover after days 0.544 dies=True fake=False.
- ds>0 leftover 0.640 T1 0.526 T2 0.466 T3 0.610.
- a_debt_service/a_in3 Y3 0.611 leftover after days 0.517. after days+size+f_ds_r 0.487 fake=False. after days+f_debt_service 0.574.
- ds>0 Y3 0.499 leftover after days+f_ds_r 0.552 after days+size 0.613 after f_ds_r 0.499 ρ vs f_ds_r 0.544 n=1924.
- ever-ds leftover 0.534 n=2999. never-ds leftover 0.667. 2025 leftover 0.489 2026 leftover 0.450.
- f_ds_r leftover after days 0.528 (quote 0.528 CONFIRM). f_debt_service leftover 0.484. f_fc_r leftover 0.449 (quote 0.449).
- last-labeled Y3 0.601 leftover 0.453 n=725. MoM Δ Y3 0.508 leftover 0.600 dies=False. after days+a_fin_cost 0.513. T3 leftover after days+f_ds_r 0.591.
- has-ds dummy Y3 0.614 leftover after days 0.473 after days+size 0.476. a_debt_service/a_op_out Y3 0.607 leftover 0.519. last-3 leftover 0.472 n=1901.
- MoM leftover after days 0.600 fake=False ρ(resid,days)=-0.762. after days+f_ds_r 0.556 after days+size 0.528. T3 leftover after days 0.610 after days+size+f_ds_r 0.607 n=2187.
- 3m rolling Y3 0.618 leftover 0.527 dies=True. ever-ds share Y3 0.613 leftover 0.524. last-labeled ever-ds Y3 — leftover — n=333.
- T3 Y3 0.635 leftover after days+size 0.619 after days+a_fin_cost 0.609 after days+size+f_ds_r+a_fin_cost 0.600 fake=False n=2187. never-ds leftover 0.667 fake=False ρ=-0.742.
- a_debt_service/a_n_tx Y3 0.609 leftover 0.482. after days+f_fc_r 0.512. after days+a_n_tx+f_ds_r 0.480. Dark last-labeled leftover — n=0.
- T3 leftover after days+a_n_tx 0.615 after days+a_op_out 0.599 after days+size+f_ds_r+a_fin_cost+a_n_tx 0.599 fake=False. T3 ds>0 Y3 — leftover — n=1146. T3 last-labeled leftover — n=252.
- within-co rank Y3 0.462 leftover 0.530 dies=True. after days+f_ds_r+a_n_tx 0.480. high-days leftover 0.604 low-days leftover 0.555.
- Bootstrap leftover-after-days rank p05=0.447 p50=0.514 p95=0.548 share<0.55=95.0% n=40.

unused leftover after days: honest rank 0.484 dies (OLS 0.631 fake=False). DROP from the 44 as Y3 X. Off the 15-col card. Do not grow TURNOVER.

## What failed / next

- none

Elapsed 14s.
