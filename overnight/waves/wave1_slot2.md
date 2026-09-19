# Wave 1 slot 2 — Family B (liquidity / balance path)

## Files written

- `analysis/features/liquidity.py` — `SOURCE_TABLES = ["balances", "transactions", "banking_products"]`, `FAMILY = "b"`, `build(con, grid)`.
- This note.

No other files. Not committed. `connect()` is read-only. Holdout is not used to fit anything (no percentiles / bins / models in this module). Features are computed for all grid companies; coverage below is **train only**.

## What it does

Reconstructs company period-end cash on `checking` + `saving` + `tpv` exactly like `score_pipeline._liquidity`: snapshot `balances` as-of 2026-09-01, then `end_bal = snapshot - sum(flows after the period)`. Snapshot-day txs stay in the walk-back (no date filter on reconstruction). Operational outflow (`out3`) is computed here from `transactions` + `CAT_MAP` (`date < 2026-09-01`); cashflow is not imported.

Monthly windows are 3 / 6 periods. Weekly grid uses 13 / 26 (runway still `liq / max(out3/3, 1)` so the unit stays months). No look-ahead past period end except the documented backward walk.

## Columns and train non-null %

Train: 21,157 company-months, 1,214 companies (holdout 72 companies / 1,073 rows left out of these %). Smoke-tested in `/tmp/wave1_slot2_smoke.py`.

| column | train non-null % | train companies with any |
|--------|------------------|--------------------------|
| `b_liq` | 98.98 | 98.43 |
| `b_runway` | 87.68 | 98.43 |
| `b_d_runway` | 70.75 | 98.19 |
| `b_neg_liq_3` | 87.68 | 98.43 |
| `b_min_liq_3` | 87.68 | 98.43 |
| `b_mean_liq_3` | 87.68 | 98.43 |
| `b_below_0` | 98.98 | 98.43 |
| `b_below_half_runway` | 87.68 | 98.43 |
| `b_bal_vol` | 70.75 | 98.19 |
| `b_neg_episodes` | 70.75 | 98.19 |

Gaps: ~1.6% of train companies have no cash-product snapshot (liq stays null). 3-month rolls need two burn-in months after first activity; 6-month rolls / `d_runway` need five.

Weekly (train, 91,560 rows): `b_liq` / `b_below_0` 99.00%; 13-week family 83.34%; 26-week family 66.39%.

## Checks that passed

- `b_liq` matches `_liquidity` (max abs 0).
- `b_runway` and `b_neg_liq_3` match Javier’s `runway` / `neg_liq` (max abs 0).
- `b_d_runway` is **not** clipped (task). After `clip(-12, 12)` it matches Javier’s `d_runway`.
- Truncating the grid at 2025-12 does not change any earlier row (outflow side has no look-ahead).
- `b_runway` stays in [-6, 24]. Train `b_below_0` rate among non-null ≈ 7.8%.

## What failed

Nothing in the smoke test.

## Next idea

Product-level min / “one empty account” (company-sum `b_liq` can hide a dry checking account). Days-below-zero needs a daily walk; this wave only has month/week ends. `b_below_half_runway` is common (~38% of train months with runway) — keep as a flag, not a rare event.
