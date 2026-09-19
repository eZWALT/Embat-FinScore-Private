# Wave 4 — a_fin_cost leftover after days

Agent `b17e9c44`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/fin_cost_qa.py`
- `analysis/outputs/fin_cost_qa.md`
- `analysis/outputs/fin_cost_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not overwrite `y9_why.*`, `fc_r_qa.*`, `ds_r_qa.*`, `catmix.py`, `util_snap_qa.*`, `ogtg_qa.*`, `n_types_qa.*`. Did not touch `debt.py`, `gbm_core.py`, the 15-col card, TURNOVER, product/, Family M, or `build_targets`. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. TURNOVER **0.720 / 0.712**.

## Locked verdict

| object | decision |
| --- | --- |
| leftover after days | **CLOSE** rank 0.483 fake=True |
| as Y3 X | **CLOSE unused leftover / DROP from the 44 as Y3 X** |
| leftover after f_fc_r | **0.634** |
| leftover after f_fin_cost | **0.635** |
| days leftover after a_fin_cost | **0.673** |
| Q6 | **CLOSE** |

Y3 native 0.634 vs days 0.711 vs size 0.617 vs f_fc_r 0.559. beat_size=0.018. ρ days 0.531 f_fin_cost 1.000 f_fc_r 0.663. lag1 leftover after days_lag1 0.469. Bootstrap leftover-after-days p05/p50/p95 0.431 / 0.472 / 0.554. Y2 leftover 0.455. f_fc_r leftover replica 0.449. Exact rewrite a==f 100.0%. log1p leftover 0.483. month-ratio leftover 0.536. Ever-fee dummy leftover after days 0.627 after days+size 0.610. High-fee dummy leftover 0.638 after days+f_fc_r 0.593. Logo min 0.477. Exact rewrite 100% a==f_fin_cost. High-fee dummy leftover after days 0.638 (not the asked amount; do not put on the 15-col card). Amount leftover after days+f_fc_r_lag3 0.555.

## What failed / next

- no replica miss; leftover after days 0.483 dies and is a fake days leak (ρ=-0.845; OLS 0.683). Exact twin of f_fin_cost ρ=1.000. beat_size +0.018 fails. CLOSE leftover / DROP from the 44. Do not put a_fin_cost on the 15-col card. KEEP f_fc_r_lag3 on TURNOVER.

Elapsed 60s.
