# Wave 4 — j_pay_match leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/pay_match_qa.py`
- `analysis/outputs/pay_match_qa.md`
- `analysis/outputs/pay_match_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `match.py`, `pending_qa.*`, `june_tx_qa.*`, `ogtg_qa.*`, `ap_open_qa.*`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Family J computed in-memory — **not merged**. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.

## Locked verdict

| object | decision |
| --- | --- |
| `j_pay_match` as Y3 X / 15-col card | **KEEP-Q5 only** |
| `j_pay_match` as engine X on the 44 | **DROP** |
| Family J merge | **do not merge J** |
| Q5 diagnostic | **KEEP-Q5** |
| `y_pay_match` | **PARK** |

Y3 leftover after days rank 0.556 (dies=False, fake=False); inverse days after pay_match 0.710. Single 0.567 vs days 0.711 vs size 0.617. ρ vs pending -0.093 vs iss 0.678 vs size 0.000. after pending 0.566. Q6 lag1 leftover 0.586. Y5 leftover after size 0.533. Dark 470 NaN=True. leftover after days rank 0.556 lives but beat-size FAIL (0.567 vs 0.617). J stays KEEP-Q5 diagnostic, not a Y3 X. Do not merge J.

## Locked extras

- Leftover after days 0.556 is thin (bootstrap p05=0.389 p50=0.570 p95=0.637; 37.5% die).
- Leftover after days+iss 0.526 dies. after days+pending+iss 0.533 dies.
- Not a pending twin (ρ=-0.093). Y5 leftover after size 0.533 dies. Quoted Y5 leftover 0.588 ρ 0.262 was d_tx after J.
- Q6 lag1 leftover after days_lag1 0.586 lives but beat-size FAIL (0.596 vs 0.617). CLOSE as 44.
- Q1 low-match Y3 10.4% vs Q5 5.9%. Q1 dummy leftover 0.554 / after days+size 0.535 dies.
- Dark 470 nn=0 CONFIRM. Do not merge J.

## What failed / next

- none

Elapsed 20s.
