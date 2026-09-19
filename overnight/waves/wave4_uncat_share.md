# Wave 4 — a_uncat_share leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/uncat_share_qa.py`
- `analysis/outputs/uncat_share_qa.md`
- `analysis/outputs/uncat_share_qa.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `uncat_qa.*`, `missing_cp_qa.*`, `d_tx_qa.*`, `top1_qa.*`, `n_types_qa.*`, `ap_issued_qa.*`, `cashflow.py`, CAT_MAP, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged. d_cust_top1 leftover 0.525 DROP stays.

## Locked verdict

| object | decision |
| --- | --- |
| `a_uncat_share` as Y3 X / 15-col card | **CLOSE unused leftover** |
| `a_uncat_share` as engine X on the 44 | **DROP** |
| uncat as health Y | **PARK** |
| amount-uncat merge | **CLOSED** |

Y3 leftover after days rank 0.573 (dies=False, fake=False); inverse days after uncat 0.712. Single 0.542 vs days 0.711 vs size 0.617. ρ vs miss_cp 0.131 vs d_tx -0.141 vs days 0.180 vs size 0.001. after miss_cp 0.533 after d_tx 0.536. ICC 0.985 trait=True. Q6 lag1 leftover 0.557. Dark nn=7206 mean 0.293 vs ERP 0.238. count↔amount ρ=0.889. leftover after days rank 0.573 lives but fails beat-size (0.542 vs 0.617). Style dummy ICC 0.985. DROP from the 44 as Y3 X. PARK as Y.

Dark 470 uncat nn=7206 CONFIRM uncat ≠ no ERP. ICC MSB/(MSB+MSW) 0.985 CONFIRM TRAIT; ANOVA ICC(1) 0.798; leftover is BETWEEN (company-mean leftover 0.586 lives / demean 0.484 dies). Permute-within-days p50=0.508 p90=0.526 — observed leftover 0.573 is above the null. Bootstrap p05/p50/p95 0.529/0.574/0.622 (22.5% die). Q1|Q5 U-shape flag Y3 0.639 leftover 0.558 is a footnote, not a 44 stem. Do not merge amount-uncat. Do not add `uncategorized` to CAT_MAP. Do not put uncat on the 15-col card. Do not invent y_uncat.

## What failed / next

- none

Elapsed 30s.
