# Wave 4 — h_sib_neg_share leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/sib_neg_qa.py`
- `analysis/outputs/sib_neg_qa.md`
- `analysis/outputs/sib_neg_qa.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `sibling_h.*`, `uncat_share_qa.*`, `n_types_qa.*`, `ap_issued_qa.*`, `groupctx.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged. Q5 sister-runway footnote **KEEP**.

## Locked verdict

| object | decision |
| --- | --- |
| `h_sib_neg_share` as Y3 X / 15-col card | **CLOSE unused leftover** |
| `h_sib_neg_share` as engine X on the 44 | **DROP** |
| H as health Y | **PARK** |
| Q5 sister mean b_runway ≥ 1 | **KEEP** |

Y3 leftover after days rank 0.453 (dies=True, fake=False); inverse days after sib_neg 0.724. Single 0.434 vs days 0.711 vs size 0.617. ρ vs group_size 0.089 vs mixed 0.025 vs days 0.063 vs size 0.046. after size 0.446 T1 residual 16.7%. mixed leftover 0.420 all-dark 0.375. Q6 lag1 leftover 0.482. share_group_in 0.638. unused leftover after days: honest rank 0.453 dies (OLS 0.453 fake=False). Single 0.434 vs days 0.711 / size 0.617. DROP from the 44 as Y3 X. PARK as Y (H marks sister existence; hidden test is new groups). Q5 sister-runway footnote KEEP.

Dark 470 = mixed 110 + all-dark 360 CONFIRM. T1 mixed−all-dark +16.7pp CONFIRM; y11 T1 +12.5pp CONFIRM. Sister runway≥1 T1 +7.0pp CONFIRM. H leftover after runway flag dies — H does not carry the footnote. Bootstrap leftover p05/p50/p95 0.404/0.452/0.543 (97.5% die). Permute-within-days p50=0.511 — observed 0.453 is below the null. Dark-only mixed dummy 0.566 CONFIRM. Hidden test is new groups. Do not put H on the 15-col card. Do not invent y_sib_neg.

## What failed / next

- none

Elapsed 30s.
