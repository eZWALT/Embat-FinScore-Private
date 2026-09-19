# Wave 4 — recency / last-tx-before-June QA

Agent `4545d7a6`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/recency_qa.py`
- `analysis/outputs/recency_qa.md`
- `analysis/outputs/recency_calendar.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `ops.py`, `y6_activity.py`, `a_vol_qa.py`, `transfer_qa.py`, `gap_sd_qa.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, or the 15-col card. Night Y3 quote stays **0.762 / 0.752**. Days 0.711 untouched.

## Locked verdict

| object | decision |
| --- | --- |
| `c_recency_days` on the 44 | **DROP from the 44** |
| `c_recency_days` as a health Y | **PARK** |
| `c_last_tx_before_2026_06` | **PARK** |
| Q6 recency lag | **CLOSE** |
| `y6_silent_60` / `created_at` | **PARK** (not revived) |

Y3 recency 0.659 leftover-after-days 0.607 vs size 0.617 (Δ -0.010) vs days 0.711. CLOSE as quiet twin of days. Log leftover 0.673 beats size 0.617 but loses to days 0.711 (busy 0.680 vs days 0.711). Not on the 15-col card.

Javier extract 61 vs as-of-Aug 62 (COMP_0981 is the +1). Flag only fires 2026-06..08 (203 CM); Y3 last labeled 2026-02-01 — never reaches the window. PARK as extract artifact; DROP from the 44.

## What we measured (train)

- 61 vs 62: extract 61 / as-of-Aug 62; COMP_0981 is the +1=True.
- Y3 recency 0.659 vs size 0.617 vs days 0.711.
- Leftover after days 0.607 (Δ size -0.010).
- Extract hole: True.
- Y3 recency now 0.659 vs lag1 0.619 (Δ 0.040); short lag1 0.623. Days lag1 replica 0.684 (short 0.684). CLOSE as Q6 — lag does not hold.

## What failed / next

- leftover after days 0.607 vs size 0.617 — CLOSE as quiet twin
- June flag is a 2026-06 extract hole — DROP from the 44 / PARK as artifact
- recency tail piles at Aug (p90 29.4 vs early 10.0; ≥30d 10.0% vs 3.4%) — extract-end still
- company-mean leftover 0.462 vs size — quiet-company twin of days
- log leftover 0.673 beats size but loses to days 0.711; busy log leftover 0.680 — CLOSE as quiet twin

Elapsed 7s. Log leftover 0.673 / busy 0.680 still loses to days 0.711. Tail piles at Aug (p90 29 vs 10).
