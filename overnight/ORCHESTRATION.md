# Night orchestration — until 2026-09-19 08:00 CEST

Parent agent owns merge, verification, next-wave prompts, and the 20-minute loop.
At most **3** child agents at once (token budget). Do not interrupt runners to
get down to 3 — wait for natural finishes. New children are **long-lived**:
stay on one owner-file subspace for ≥30 minutes (write, run, fix, next
experiment). Disjoint file owners. Iterative waves, not one shot.

## Wave 0 (parent, done before children)

Wipe invalid 16h search. Rebuild `overnight/` as status. Skeleton under `analysis/`.
Rebuild `data/embat.duckdb`. Contracts in `CONTRACT.md`.

## Wave 1 — build X and first Ys (8 slots)

| slot | owner file(s) | subspace |
|------|----------------|----------|
| 1 | `analysis/features/cashflow.py` | Family A |
| 2 | `analysis/features/liquidity.py` | Family B |
| 3 | `analysis/features/invoices.py` | Family E |
| 4 | `analysis/features/debt.py` | Family F |
| 5 | `analysis/features/ops.py` | Family C |
| 6 | `analysis/features/counterparties.py` | Family D |
| 7 | `analysis/targets/y1_forecast.py`, `y2_stress.py` | Y1 + Y2 |
| 8 | `analysis/targets/y5_payment.py`, `analysis/evaluate/protocol.py` | Y5 + protocol |

Parent verifies: import `build`, prefix, merge without clash, coverage, no holdout in any fit.

## Wave 2 — evaluate and baselines (spawn after wave 1 passes)

Typical slots: `feature_report.py`, `baselines.py`, `build_targets.py`, Y3/Y4, Y6, family G, family H, literature scan + new Y idea.

## Wave 3+ — models

GBM panel, weekly TS+exog sample, per-group if clusters exist, raise/kill Ys that fail acceptance.

## Stop a direction

If a family or Y fails verification or loses to baselines twice, park it and try another from the plan (Y7, Y8, weekly SARIMAX, Tobit decision).

## Loop

Every 20 minutes until 08:00: re-read `overnight/NORTH_STAR.md`, then
`overnight/LIVE.json` and `overnight/QUEUE.md`. Verify finished slots. Spawn the
next queued item **only if fewer than 3 agents are running**. Never interrupt a
runner to make room — queue instead. One new spawn per tick max. Each new
child must be scoped to iterate ≥30 minutes on its files. Lanes: data / models /
feature engineering / explainability. Do not touch `product/`. Do not stop
because a wave "looks done".
