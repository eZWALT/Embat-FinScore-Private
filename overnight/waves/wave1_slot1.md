# Wave 1 / slot 1 — Family A (cash flow)

## Files

- `analysis/features/cashflow.py` (owned). `SOURCE_TABLES = ["transactions"]`, `FAMILY = "a"`, `build(con, grid)`.
- This note. Smoke script only in `/tmp/wave1_slot1_smoke.py` (not in the repo).
- No other files edited. No commit. No 0–100 score.

## Columns (all `a_`)

Monthly levels use txs with `date` in `[period, period + 1 month)` (month-start `period`). Outflow groups negated like `score_pipeline._monthly_flows`. `CAT_MAP` from `analysis.features.common`. Holdout not used to fit anything (no refs).

| column | formula / note |
|--------|----------------|
| `a_op_in` | sum(amount \| op_in) |
| `a_op_out` | -sum(amount \| op_out) |
| `a_net` | `a_op_in - a_op_out` |
| `a_fin_cost` | -sum(amount \| fin_cost) |
| `a_debt_service` | -sum(amount \| debt_service) |
| `a_n_tx` | count |
| `a_transfer` | signed net of `transfer` |
| `a_invest` | signed net of invest deploy+return |
| `a_in3` / `a_out3` | rolling 3m sum of in / out (`min_periods=3`) |
| `a_in6` / `a_out6` | rolling 6m |
| `a_in12` / `a_out12` | rolling 12m; NaN until month 12 |
| `a_net_margin` | `(in3-out3)/in3` clip [-1,1]; **-1 if in3==0** |
| `a_io_ratio` | `in3/max(out3,1)` cap 3 (pipeline `coverage`) |
| `a_growth_3` | `in3/in3_{t-3}-1` clip [-1,1] if lag>0 |
| `a_growth_12` | YoY of the 3m inflow window: `in3/in3_{t-12}-1` |
| `a_uncat_share` | share of txs `uncategorized` or not in `CAT_MAP` |
| `a_pending_share` | `pending/(pending+booked)`; `status` exists |

Empty company-months on the grid: levels 0, shares NaN. Checked vs `score_pipeline` on train: `op_in/out/fin_cost/debt_service/n_tx/in3/out3/net_margin/coverage/growth` match (max abs ~1e-11). August 2026 `a_n_tx` excludes 9,242 txs on 2026-09-01 (0 mismatches).

## Train coverage

Train = 21,157 company-months / 1,214 companies (holdout 1,073 / 72 excluded from these %). Panel shape 22,230 × 22.

| column | train non-null % |
|--------|------------------|
| a_op_in | 100.00 |
| a_op_out | 100.00 |
| a_net | 100.00 |
| a_fin_cost | 100.00 |
| a_debt_service | 100.00 |
| a_n_tx | 100.00 |
| a_transfer | 100.00 |
| a_invest | 100.00 |
| a_uncat_share | 95.80 |
| a_pending_share | 95.06 |
| a_in3 | 88.52 |
| a_out3 | 88.52 |
| a_net_margin | 88.52 |
| a_io_ratio | 88.52 |
| a_in6 | 71.32 |
| a_out6 | 71.32 |
| a_growth_3 | 63.96 |
| a_in12 | 41.04 |
| a_out12 | 41.04 |
| a_growth_12 | 26.47 |

## Issues

- Only unmapped category is `uncategorized` (~635k txs, ~25%). Mean train `a_uncat_share` ≈ 0.26, so operating levels miss a lot of cash.
- 12m sums 41% non-null; YoY growth 26.5% (needs month 15+ and `in3_{t-12}>0`).
- `a_pending_share` mean ≈ 0.0016 (6,579 pending / 2.56M) — near-zero variance.
- 24 `is_extreme` rows kept (not dropped). Most are uncategorized; a few mapped `collection` / `cash_settlement` / `cash_withdrawal` can still spike `a_op_in`/`a_op_out`.
- Monthly only. Week-start periods would not match `date_trunc('month')`.
- `a_transfer` / `a_invest` are signed nets (not magnitudes).

## Next idea

Add `a_uncat_net` (signed uncategorized flow, maybe split by sign) so the 25% unmapped cash is usable as X; optional amount-weighted uncat share and `a_extreme_share` from the clean flag. Weekly analog can wait for the week grid.
