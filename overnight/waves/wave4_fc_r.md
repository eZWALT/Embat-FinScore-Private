# Wave 4 — f_fc_r leftover

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/fc_r_qa.py`
- `analysis/outputs/fc_r_qa.md`
- `analysis/outputs/fc_r_leftover.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `gbm_y7_core.py`, `gbm_core.py`, `dso_qa.py`, `cust_hhi_qa.py`, `d_tx_qa.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, LIVE, CONTEXT, canvas, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.

## Locked verdict

| object | decision |
| --- | --- |
| contemporaneous `f_fc_r` on the 44 | **DROP** |
| `f_fc_r_lag3` on TURNOVER | **KEEP** |
| F lag3 Q6 | **CLOSE** |
| Y9 identity | **fee_r raw material (Y9 forbids F as X); not a binary twin (ρ 0.13)** |

Y3 leftover after `f_ds_r` rank 0.464 (dies=True); after days 0.449. Not a twin of ds_r (ρ=0.205, R²=0.005). Y7 leftover issued_lag1 0.563 TURN_NOFC 0.557 — CLOSE add-on. Y9 ρ=0.132 Jaccard=0.113 (raw material, not the binary Y). Q6 empty-until-6 CONFIRM=True. Y3 single 0.559 vs size 0.617 vs days 0.711. A-trail recon ρ=1.000. Demean leftover after days lives but loses to size.

## What failed / next

- OLS leftover after days 0.648 is a fake leak; honest rank 0.449 dies
- Y3 single without fold 2 is 0.502 — the 0.559 mean is one-fold leftover

Elapsed 9s.
