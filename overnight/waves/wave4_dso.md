# Wave 4 — DSO leftover

Agent `e8b2c0d4`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/dso_qa.py`
- `analysis/outputs/dso_qa.md`
- `analysis/outputs/dso_leftover.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `invoices.py`, `dpo_qa.py`, `delay_qa.py`, `gbm_y7_core.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, LIVE, canvas, TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Do not put DSO back. Night Y3 stays **0.762 / 0.752**. Days 0.711 untouched.

## Locked verdict

| object | decision |
| --- | --- |
| DSO as Y7 leftover after issued_lag1 | **CLOSE** |
| DSO as TURNOVER stem / put DSO back | **CLOSE** (do not grow 0.720) |
| DSO as Y3 X / the 44 | **CLOSE / DROP from the 44** |
| DSO leftover after delay_coll | **not a delay twin; leftover-after-delay is a |DSO|>24 tail** |
| DSO as a health Y | **PARK** |
| Q6 DSO on short books | **CLOSE** |
| e_dso_proxy on the 44 | **DROP** |

ρ DSO vs DPO 0.456 vs delay 0.214 vs size 0.064. Y7 leftover after issued_lag1 0.452 / delay 0.561. Y3 DSO 0.564 leftover-days 0.474 vs days 0.711 (drop>|24| leftover 0.424 — not only tail). Q6 short lag1 0.422 early cov 87.3%. Fold 4 DSO 0.342 vs issued 0.647. Short-DSO Q1 univariate 0.455 (B_shallow OOF 0.410 locked). Y7 leftover-after-delay 0.561 is a |DSO|>24 tail (drop 0.424 clip24 0.404). log1p leftover Y3 0.560 Y7-iss 0.431. lag1 vs days 0.023 (unclipped leftover-vs-days ρ was a false clone). Long so-far leftover-iss 0.613 vs issued 0.646 — slice only, does not flip DROP. Rolling Y7 t+3 leftover 0.489 issued 0.596. Rolling Y3 t+3 leftover — days —. |DSO|>24 persist twice 126 chronic-half 27. Pooled Y3 t+3 leftover 0.728 days 0.633. Rolling leftover-delay 0.521. Clean leftover-iss 0.400. Clean-refit leftover-days 0.423 vs days 0.713 size 0.622. Pooled Y3 leftover 0.728 drop>|24| 0.666 not KEEP. |DSO|>24 month min 0.0% max 15.5% spike 2026-08-01. Drop-Aug leftover-iss 0.452 leftover-delay 0.561. Y6 leftover-iss 0.584 (not a new Y). Labeled-slope leftover Y3 0.697 Y7-iss 0.452. Rank leftover-days 0.569 leftover-iss 0.431. Issued+days leftover 0.691 ρ=-0.974. Clip leftover issued+days 0.547 ρ=0.107. Labeled leftover-vs-days ρ=-0.961 FALSE clone. Mix leftover-iss 0.448.

## What failed / next

- Y3 leftover after days 0.474 — CLOSE / DROP from the 44
- Y7 leftover dies after issued_lag1 0.452
- DROP e_dso_proxy from the 44
- Q6 CLOSE — exists early (unlike delay; early6 Y7 finite 87.3%) but Y7 lag1 short 0.422 dies — stock/flow is not a TURNOVER lead
- do not put DSO back on TURNOVER; night quote stays 0.720 / 0.712
- CONFIRM issued owns fold 4 (0.647 vs DSO 0.342)
- CONFIRM short-DSO Q1 hole univariate 0.455 (B_shallow OOF 0.410 locked; n=1331)
- Y3 leftover after days drop-tail 0.424 (raw 0.474) — not only a tail / still loses to days
- SHAP #1 not a delay twin; leftover-after-delay is a |DSO|>24 tail: leftover after delay 0.561 same-n 0.561; delay leftover KEEP 0.581
- Y3 Q6 lag1 0.588 short 0.602
- Y3 lag1 leftover-days 0.711 drop>24 0.553
- clip24 ICC 0.912 — do not write clip to store
- Y7 leftover-after-delay 0.561 is a |DSO|>24 tail (drop 0.424 clip24 0.404)
- Y3 leftover days+size 0.618 drop>24 0.418 (days-only leftover 0.474)
- Y3 lag1 leftover vs days ρ=-0.984 raw-lag1-days 0.023 FALSE clone (tail rank) leftover 0.711
- fat-issued leftover Y3 0.460 vs days 0.684
- chronic-85 Y3 DSO 0.564 → 0.566
- log1p leftover Y3-days 0.560 Y7-iss 0.431 Y7-delay 0.389
- DSO vs open ρ=0.600 Y7 open 0.465 open-leftover-iss 0.596
- lag1 vs days ρ=0.023 clip24-lag1 vs days 0.023 clip24-lag1 leftover 0.569
- Y3 leftover-after-issued 0.560 drop>24 0.416 tail
- max DSO 2190260.7 n>1000=70 n>1e5=5
- demean leftover-iss 0.461 mean leftover-iss 0.518
- Y3 fold leftover-days 0.474 fold4 0.299
- Q1 leftover-iss 0.472 leftover-delay 0.521 issued 0.604
- after7 leftover-delay 0.561 drop 0.424 leftover-iss 0.372
- long so-far DSO 0.607 leftover-iss 0.613 issued 0.646 short leftover-iss 0.503
- leftover after od30 Y7 0.477 clip 0.549 Y3 0.374
- long leftover fold-wise 0.613 n=828 pos=157 vs issued 0.646 — slice only
- rolling Y7 t+3 leftover-iss 0.489 DSO 0.485 issued 0.596
- rolling Y3 t+3 leftover-days — DSO — days —
- |DSO|>24 persist once 213 twice 126 chronic-half 27
- pooled Y3 t+3 leftover 0.728 days 0.633 n=1301 pos=98
- rolling Y7 t+3 leftover-delay 0.521 delay 0.510 DSO 0.485
- clean |DSO|≤24 leftover-iss 0.400 leftover-days 0.654 vs issued 0.555 days 0.713
- clean-refit leftover-days 0.423 DSO 0.430 days 0.713 size 0.622 leftover-iss 0.420
- pooled Y3 robust leftover 0.728 drop>|24| 0.666 DSO 0.733 days 0.633 size 0.584 not KEEP
- |DSO|>24 by month min 0.0% max 15.5% spike 2026-08-01
- drop-Aug leftover-iss 0.452 leftover-delay 0.561 leftover-days 0.582
- Y6 leftover-iss 0.584 DSO 0.586 issued 0.725 — not a new Y
- labeled-slope leftover Y3-days 0.697 Y7-iss 0.452 Y7-delay 0.565
- rank(DSO) leftover-days 0.569 leftover-iss 0.431 leftover-delay 0.382 raw-rank Y3 0.564
- labeled leftover vs days ρ=-0.961 raw -0.012 FALSE clone leftover 0.697 clip 0.458
- mix leftover-iss 0.448 DSO leftover 0.452 Y3 mix leftover-days 0.692
- holdout month coverage nn=450 (no AUROC)
- mix leftover vs days ρ=-0.983 leftover 0.692 FALSE clone
- leftover pending Y7 0.431 Y3 0.530 pending+days 0.530 AP Y7 0.449
- leftover delay_paid Y7 0.471 Y3 0.376 zero_in Y7 0.392 f_ds_r Y7 0.424
- zero_in leftover vs days ρ=-0.058 leftover 0.574 not a clone
- leftover zero_in+days 0.624 vs days 0.711
- zero_in+days leftover vs days ρ=0.746 leftover 0.624 not a clone
- leftover issued+days 0.691 vs days 0.711 ρ=-0.974
- clip leftover issued+days 0.547 ρ=0.107 vs days 0.711
- leftover f_ds_r+days 0.675 ρ=-0.937 vs days 0.711
- leftover delay_paid+days 0.624 ρ=0.702 vs days 0.711
- leftover AP+days 0.529 ρ=0.010 vs days 0.711
- leftover DPO+days 0.647 ρ=0.807 vs days 0.711
- leftover overdue+days 0.541 ρ=0.121 vs days 0.711
- clip leftover delay_paid+days 0.586 ρ=0.101 vs days 0.711
- clip leftover DPO+days 0.555 ρ=0.105 not a clone vs days 0.711
- leftover open+days 0.377 ρ=0.671 vs days 0.711
- leftover delay+days 0.647 ρ=0.878 vs days 0.711
- clip leftover delay+days 0.608 ρ=0.027 not a clone vs days 0.711
- holdout SIZE cov T1 33.9% T2 56.3% T3 37.7%
- leftover open+issued Y7 0.453 ρ=0.219 vs issued 0.630
- leftover pending+issued Y7 0.432 ρ=0.031 vs issued 0.630
- leftover AP+issued Y7 0.449 ρ=0.041 vs issued 0.630
- clip leftover f_ds_r+days 0.547 ρ=0.117 not a clone vs days 0.711
- leftover DPO+issued Y7 0.499 ρ=0.150 vs issued 0.630
- clip leftover zero_in+days 0.560 ρ=0.083 not a clone vs days 0.711
- leftover overdue+issued Y7 0.521 ρ=0.079 vs issued 0.630
- leftover delay_paid+issued Y7 0.536 ρ=-0.002 vs issued 0.630
- leftover size+issued Y7 0.463 ρ=-0.512 vs issued 0.630
- leftover f_ds_r+issued Y7 0.414 ρ=0.220 vs issued 0.630
- leftover AP issued+days 0.475 ρ=0.670 vs days 0.711
- leftover od30+issued Y7 0.482 ρ=-0.031 vs issued 0.630
- leftover AP issued+issued Y7 0.452 ρ=0.231 vs issued 0.630
- leftover zero_in+issued Y7 0.506 ρ=0.194 vs issued 0.630

Elapsed 13s.
