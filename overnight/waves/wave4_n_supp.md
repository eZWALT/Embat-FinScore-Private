# Wave 4 — d_n_supp leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/n_supp_qa.py`
- `analysis/outputs/n_supp_qa.md`
- `analysis/outputs/n_supp_qa.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `supp_hhi_qa.*`, `n_cust_qa.*`, `n_types_qa.*`, `ap_issued_qa.*`, `cust_hhi_qa.*`, `counterparties.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged. Y5 leftover after size is report-only.

## Locked verdict

| object | decision |
| --- | --- |
| `d_n_supp` as Y3 X / 15-col card | **DROP from the 44 as Y3 X** |
| `d_n_supp` as engine X on the 44 | **DROP** |
| `y_n_supp` | **PARK** |

Y3 leftover after days rank 0.587 (dies=False, fake=False); inverse days after n_supp 0.666. Single 0.699 vs days 0.711 vs size 0.617. ρ vs n_cust 0.632 vs top1 -0.661 vs HHI -0.726 vs days 0.607 vs size 0.546. after top1 0.639 after HHI 0.626 after n_cust 0.649. Q6 lag1 leftover 0.598. Y5 leftover after size 0.561. vs n_cust same=False. Leftover after size 0.657 lives (not a size clone); leftover after days+top1 dies. Permute leftover p50=0.542 — observed 0.587 sits above the null. leftover after days rank 0.587 lives, beat-size PASS, not a twin, but SIZE (ρ vs log1p(a_in3)=0.546 ≥0.50). KEEP-as-X fails the SIZE gate. Off the 15-col card. Do not invent y_n_supp.

## What failed / next



Elapsed 41s.
