# Wave 4 — f_outstanding_gt_granted leftover after days

Agent `b17e9c44`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/ogtg_qa.py`
- `analysis/outputs/ogtg_qa.md`
- `analysis/outputs/ogtg_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not overwrite `factoring_qa.*`, `n_types_qa.*`, `ds_r_qa.*`, `fc_r_qa.*`, `debt_schedule_qa`, `sib_neg_qa.*`, `ap_issued_qa.*`. Did not touch `debt.py`, `gbm_core.py`, the 15-col card, TURNOVER, product/, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. TURNOVER **0.720 / 0.712**.

## Locked verdict

| object | decision |
| --- | --- |
| leftover after days | **CLOSE** hole 0.711 native n_pos=0 |
| as Y3 X | **PARK as Y / DROP from the 44 as Y3 X** |
| last-month-only | **True** cov 5.7% |
| leftover after n_types | **0.578** |
| days leftover after hole | **0.711** |
| extract-hole | **True** n1=32 |
| Q6 | **CLOSE** |

Y3 native — hole-fillna0 0.500 vs days 0.711 vs size 0.617 vs n_types 0.578. ρ days 0.085 types 0.278 util 0.152. lag1 leftover after days_lag1 0.632. Trait leftover after days 0.684 fake=True. Bootstrap trait p05/p50/p95 0.656 / 0.688 / 0.718. Last-month dummy leftover 0.711 fake=True. Trait leftover after n_types 0.545 boot-n_types p50 0.548. Random 32 dummy leftover p50 0.693 obs 0.684. OGTG leftover after LOC dummy 0.535. Last-month days+n_types leftover 0.540 boot p50 0.574. Last-month days+n_types+has_loc leftover 0.538. Q6 lag1 after days 0.500. 1/24 months defined. Ever-Y3 recover 3/32=12.5% vs 24.4%.

## What failed / next

- no replica miss; native leftover undefined (Y3 n_pos=0 on 2026-08). fillna0 / last-month dummy leftover 0.711 is a fake days leak (ρ=-0.996). Trait leftover 0.684 is also a fake days leak (ρ=-0.915). Honest leftover after last-month days+n_types 0.540 dies; after LOC dummy 0.535 dies. PARK snapshot X / DROP from the 44.

Elapsed 51s.
