# Wave 2 slot 2 — Family H (group context)

## Files written

- `analysis/features/groupctx.py` — `SOURCE_TABLES=["companies","groups","transactions"]`, `FAMILY="h"`, `build(con, grid) -> company_id, period, h_*`
- This note

Assembler `company_meta` already carries `group_id` / `group_size`. This module still emits period-level H columns. Does not import `cashflow.py` or `liquidity.py`; sibling op flows are aggregated from `transactions` with `CAT_MAP` (same op_in / op_out groups as Family A). No look-ahead: `date_trunc` to the grid grain and `date <= period end`. Holdout not used to fit (formulas only). Split is group-aware (0 mixed groups).

Smoke: `/tmp/smoke_h_groupctx.py` → `/tmp/h_groupctx_smoke.json` (22,230 rows, 1,286 companies; leave-one-out identity, solo rules, verify_wave prefix OK).

## Columns

| column | meaning |
|--------|---------|
| `h_group_size` | static `groups.n_companies_in_sample` |
| `h_n_siblings_active` | other group companies with ≥1 tx this period |
| `h_sib_in` / `h_sib_out` / `h_sib_net` | sibling op_in, op_out, in−out (company excluded) |
| `h_share_group_in` | company op_in / (company + sibling op_in) if denom > 0 |
| `h_sib_neg_share` | share of siblings with monthly net < 0, denom = size−1 |

Solo groups (size 1): sibling sums and active count = 0; share = 1 if company op_in > 0 else NaN; `h_sib_neg_share` NaN.

## Train coverage (1,214 companies / 21,157 company-months; holdout 72 excluded)

| column | train non-null % |
|--------|------------------|
| `h_group_size` | 100.00 |
| `h_n_siblings_active` | 100.00 |
| `h_sib_in` | 100.00 |
| `h_sib_out` | 100.00 |
| `h_sib_net` | 100.00 |
| `h_share_group_in` | 97.98 |
| `h_sib_neg_share` | 94.90 |

## What failed

Nothing material. `h_share_group_in` is null when the group has no operational inflow that month (~2.0% of train company-months). `h_sib_neg_share` is null for solo groups (71 groups / 71 companies; ~5.1% of train company-months). 9,242 txs on 2026-09-01 stay out of August rows.

## Next idea

Trailing 3-month sibling net / sibling-neg persistence (contagion that is not a single noisy month). Do not pull Family B into this file; if sibling runway is wanted, join B on the assembled store in a later slot.
