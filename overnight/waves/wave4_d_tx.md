# Wave 4 — d_tx leftover

Agent `7f2e91c4`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/d_tx_qa.py`
- `analysis/outputs/d_tx_qa.md`
- `analysis/outputs/d_tx_leftover.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `counterparties.py`, `missing_cp_qa.py`, `y5_why.py`, `dso_qa.py`, `brief_map.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, LIVE, CONTEXT, canvas, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER 0.720 / 0.712. Y5 0.611 quote stands.

## Locked verdict

| object | decision |
| --- | --- |
| d_tx as Y3 X | **CLOSE / DROP from the 44** |
| d_tx as Y5 X | **PARK as X** |
| Night Y5 0.611 quote | **KEEP the 0.611 quote** |
| d_tx as a health Y | **PARK** |
| Q6 | **CLOSE** |
| d_tx_cp_share on the 44 | **DROP** |

ρ vs miss_cp -0.947 CONFIRM twin. Y3 leftover-days OLS 0.600 rank 0.537 labeled 0.534 vs days 0.711. Y5 leftover-size 0.579 after miss 0.454. Fold-3 hole 85.9% drop-fold3 train 0.541 drop-4-groups train 0.535 collapse=True. J leftover Y5 0.588 ρ=0.262 (not merged).

## What failed / next

- Y3 leftover after days OLS 0.600 rank 0.537 — CLOSE / DROP from the 44
- leftover after miss_cp dies Y3 0.509 Y5 0.454
- fold-3 one-group hole CONFIRM — CLOSE as leave-one-group law
- DROP d_tx_cp_share from the 44
- Q6 CLOSE short lag1 0.431 rank leftover 0.532
- drop 4 hole groups ['GROUP_0079', 'GROUP_0132', 'GROUP_0081', 'GROUP_0018']: Y5 train 0.535 (same collapse as drop fold 3)
- Y3 leftover after J 0.602 Y5 0.588 ρ=0.262 — do not merge J

Elapsed 12s.
