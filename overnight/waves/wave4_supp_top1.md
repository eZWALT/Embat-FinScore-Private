# Wave 4 — d_supp_top1 leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/supp_top1_qa.py`
- `analysis/outputs/supp_top1_qa.md`
- `analysis/outputs/supp_top1_qa.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `supp_hhi_qa.*`, `top1_qa.*`, `n_supp_qa.*`, `sib_neg_qa.*`, `ogtg_qa.*`, `ap_issued_qa.*`, `counterparties.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged. Y5 protective-tail footnote **KEEP**.

## Locked verdict

| object | decision |
| --- | --- |
| `d_supp_top1` as Y3 X / 15-col card | **CLOSE unused leftover** |
| `d_supp_top1` as engine X on the 44 | **DROP** |
| Y5 protective tail | **KEEP** |
| `y_supp_top1` | **PARK** |

Y3 leftover after days rank 0.429 (dies=True, fake=False); inverse days after supp_top1 0.718. Single 0.641 vs days 0.711 vs size 0.617. ρ vs HHI 0.987 vs n_supp -0.661 vs cust_top1 0.117 vs size -0.323. after HHI OLS 0.555 is the rank leftover; rewrite leftover dies on OLS (see extras). after n_supp 0.396. Q6 lag1 leftover 0.478. Y5 tail 2.7% vs 8.6%. vs cust same=False. unused leftover after days: honest rank 0.429 dies (OLS 0.427 fake=False). Also TWIN of ['d_supp_hhi']. DROP from the 44 as Y3 X. Y5 protective-tail footnote KEEP. Do not invent y_supp_top1. Off the 15-col card.

## Locked extras

- Dark 470 nn=0 CONFIRM. Calendar incomplete nn=0 CONFIRM.
- vs cust_top1 ρ 0.117 different object.
- HHI rewrite leftover OLS 0.403 dies / rank 0.555 chance (R²=0.955).
- Bootstrap leftover-after-days 0.363 / 0.438 / 0.600 (72.5% die).
- Y5 leftover after size 0.473 dies. HHI>0.975 ⊂ top1>0.975 (both=73 only_HHI=0 only_top1=21). Footnote KEEP.
- n_supp>=3 leftover after days 0.381 dies harder.
- top1>0.975 dummy leftover after days 0.712 lives but dies after HHI-tail. Footnote, not a 44 stem.

## What failed / next

- none

Elapsed 38s.
