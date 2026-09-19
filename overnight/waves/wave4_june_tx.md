# Wave 4 — c_last_tx_before_2026_06 leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/june_tx_qa.py`
- `analysis/outputs/june_tx_qa.md`
- `analysis/outputs/june_tx_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `recency_qa.*`, `gap_sd_qa.*`, `ops.py`, `ogtg_qa.*`, `ap_open_qa.*`, `supp_top1_qa.*`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged. Do not revive Y6.

## Locked verdict

| object | decision |
| --- | --- |
| `c_last_tx_before_2026_06` as Y3 X / 15-col card | **PARK as extract** |
| `c_last_tx_before_2026_06` as engine X on the 44 | **DROP** |
| `y_june` / Y6 | **PARK** |

Y3 leftover after days rank 0.711 (dies=True, const=False); inverse days after flag 0.711. Single 0.500 vs days 0.711 vs size 0.617 vs recency 0.659. extract_hole=True only 2026-06..08=True. Javier 61 vs as-of 62; COMP_0981 +1=True. asof62 leftover after days 0.619. leftover after recency 0.659. Q6 lag1 0.684. extract hole: flag only 2026-06..08; last Y3 2026-02-01; mean(flag|Y3)=0.0%. Contemporaneous leftover after days rank 0.711 const=False dies. Single 0.500 (peek 0.500). Javier 61 vs 62 / COMP_0981 +1. DROP from the 44 as Y3 X. Off the 15-col card. Do not revive Y6.

## Locked extras

- Honest leftover on Y3-labeled rows: rank 0.500 const (bootstrap 0.500 / 0.500 / 0.500).
- Panel leftover 0.711 is a fake days leak (flag=0 on Y3 rows; residual is days).
- Recency leftover OLS 0.607 CONFIRM / rank 0.567 honest (not overwritten).
- Flag is not a recency rewrite (ρ=0.186).
- asof62 roster Y3 0.557 leftover 0.619 lives but beat-size FAIL. Drop COMP_0981 leftover 0.622. Bootstrap p50=0.627. Extract, not a 44 stem.
- asof train 61 have Y3=1: 24; last booking after last Y3=1: 15. Consequence, not lead.
- Jaccard(asof, recency>60 | Y3-lab)=0.088. Do not revive Y6.
- Holdout flag=1 n=7 / 3 co / 1 as-of company. Coverage only.

## What failed / next

- none

Elapsed 39s.
