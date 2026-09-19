# Wave 4 — d_n_cust leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/n_cust_qa.py`
- `analysis/outputs/n_cust_qa.md`
- `analysis/outputs/n_cust_qa.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `cust_hhi_qa.*`, `supp_hhi_qa.*`, `tax_month_qa.*`, `issued_qa.*`, `in3_qa.*`, `ss_qa.*`, `salary_month_qa.*`, `counterparties.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.

## Locked verdict

| object | decision |
| --- | --- |
| `d_n_cust` as Y3 X / 15-col card | **CLOSE unused leftover** |
| `d_n_cust` as engine X on the 44 | **DROP** |
| `y_n_cust` | **PARK** |

Y3 leftover after days rank 0.545 (dies=True, fake=False); inverse days after n_cust 0.699. Single 0.653 vs days 0.711 vs size 0.617. ρ vs HHI -0.837 vs top1 -0.806 vs days 0.507 vs size 0.381. after top1 0.456 after HHI 0.459. Q6 lag1 leftover 0.539. Y4 leftover after top1_lag3 0.530. after days+top1 0.564 is sample-shift (drop n=0); leftover after days on that n dies. T1 leftover 0.557 is fold-noisy (2/5 folds die). Permute-within-days leftover p50=0.533 — observed 0.545 sits inside the null. unused leftover after days: honest rank 0.545 dies (OLS 0.648 fake=False almost=True). Also TWIN of ['d_cust_top1', 'd_cust_hhi']. DROP from the 44 as Y3 X. Do not invent y_n_cust. Off the 15-col card.

## What failed / next



Elapsed 34s.
