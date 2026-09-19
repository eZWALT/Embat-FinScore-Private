# Wave 4 — DPO leftover

Agent `a19d4e07`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/dpo_qa.py`
- `analysis/outputs/dpo_qa.md`
- `analysis/outputs/dpo_vs_dso.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `invoices.py`, `delay_qa.py`, `gbm_y7_core.py`, `pending_qa.py`, `credit_note_qa.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. TURNDPO 0.723 is not a KEEP. Night Y3 stays **0.762 / 0.752**. Days 0.711 untouched.

## Locked verdict

| object | decision |
| --- | --- |
| DPO as Y7 leftover | **CLOSE** |
| DPO as Y7 add-on / TURNDPO | **CLOSE** (do not grow TURNOVER) |
| DPO as Y3 X / the 44 | **CLOSE / DROP from the 44** |
| DPO as Y5 X | **FORBIDDEN (Y5 never E)** |
| DPO as a health Y | **PARK** |
| Q6 DPO on short books | **CLOSE** |
| e_dpo_proxy on the 44 | **DROP** |

ρ DPO vs DSO 0.456 vs size -0.025 vs od30 0.632. Y7 leftover after DSO 0.471 / issued_lag1 0.443 / both 0.463. Y3 DPO 0.627 leftover-days 0.653 vs days 0.711 (drop>|24| leftover 0.432 — tail). Y5 leak single 0.445 leftover-paid 0.574. Q6 short lag1 0.450 early cov 93.7% (exists early unlike delay; Y7 lag dies). Fold 4 DPO 0.403 vs DSO 0.342 vs issued 0.647. Fat-issued leftover Y3 0.662. Clip24 ICC 0.908 BETWEEN.

## What failed / next

- Y7 leftover dies DSO 0.471 / issued 0.443
- DROP e_dpo_proxy from the 44
- Q6 CLOSE — exists early (unlike delay; early6 Y7 finite 93.7%) but Y7 lag1 short 0.450 dies — stock/flow is not a lead
- do not grow TURNOVER; TURNDPO 0.723 is not KEEP
- Y5 never E as X
- Y3 leftover after days is a tail/thin-issued artifact (drop>24 0.432 clip12 0.584 after-issued 0.597)
- Y3 Q6 lag1 0.610 short 0.612
- Y3 lag1 leftover-days 0.665 drop>24 0.433
- Y5 leftover drop>24 0.527 — still never E
- clip24 ICC 0.908 BETWEEN — do not write clip to store
- T1 leftover drop>24 0.706 vs days 0.615
- fat-issued leftover Y3 0.662 vs days 0.718

Elapsed 7s.
