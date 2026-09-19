# Neon staging validation (2026-09-19)

Project: `embat-health-sentinel-staging` (`sparkling-glitter-12146358`)
Region: `aws-eu-central-1`
Branch: `staging` (`br-winter-cake-b2m43sob`)
Plan: Neon **Free** ($0). Storage cap 0.5 GB.

## Sources

| Source | Role | Loaded |
|---|---|---|
| `embat_clean.duckdb` (210 MB, schema `clean`) | cleaned source | tables created; facts not loaded |
| `bundle/` schema 1.1.0 | computed score/monitor | yes |
| `bundle.zip`, `embat_clean.duckdb.zip` | compressed duplicates | **no** |
| `types.ts` | TypeScript contract copy | **no** |

DuckDB row counts: groups 250, companies 1,286, banking_products 5,987, debt_products 2,239, debt_schedule 87, balances 7,996, invoices 896,711, transactions 2,556,068, dq_log 58.

Key checks on DuckDB:

- `banking_products ∩ debt_products` on `product_id` = 0 → `core.products(product_id)` unique is valid.
- `counterparty_id` matching `company_id` = 0 (do not FK counterparties to companies).
- 1,313 transactions reference product ids absent from banking/debt (`family = 'unknown'` when facts are loaded).
- `transaction_id` and `(company_id, operation_id)` unique in this snapshot; surrogate PKs still used.

## Analytics counts (destination = origin)

| Entity | Origin | Neon |
|---|---|---|
| companies (index / profiles / dashboard view) | 1,286 | 1,286 |
| groups | 250 | 250 |
| company-months | 19,658 | 19,658 |
| alerts | 2,365 | 2,365 |
| clusters | 4 | 4 |
| score categories | 98,290 | 98,290 |
| score items | 165,596 | 165,596 |
| score reasons (level+change) | 76,273 | 76,273 |
| group members | 1,286 | 1,286 |

Differences vs 1,286 companies:

- `company_cluster` / `forecasts` = **1,282**. Four companies have a trail too short for cluster (<6 months) or forecast (<4 scored months), matching `DATA_CONTRACT.md`.

Sample: `COMP_0001` latest score **66.04**, trajectory `stable` (matches `bundle/companies/COMP_0001.json`). `COMP_1286` guard `fading`, score 50 (cap). Index↔latest month orphans: **0**. Scores outside 0–100: **0** (min 19.15, max 100).

## Size

`pg_database_size` = **83 MB**.

Largest relations: `score_items` 21 MB, `score_reasons` 20 MB, `score_categories` 11 MB, `control_charts` 8.5 MB, `company_scores` 6.2 MB.

## Plans

`EXPLAIN ANALYZE SELECT company_id, score FROM api.dashboard_companies ORDER BY score DESC LIMIT 20` → ~**32 ms**, 1,286 rows produced then top-N. `api.score_series` for `COMP_0001` → **0.07 ms**, index on `(run_id, company_id, month)`.

## Next.js

`pnpm lint`, `tsc --noEmit`, `pnpm build` succeeded with `DATABASE_URL` in `.env.local` (gitignored). `/` prerendered against Neon (1,286 companies). No test suite is defined in `product/web`.

## Core load blockers (need confirmation)

1. **TCP 5432** from this machine to Neon times out; HTTP driver works. `load.sh` (`\\copy`) needs an unblocked network or CI.
2. **Storage:** full transactions+invoices will exceed Free 0.5 GB. Upgrade org `Ruben` (`org-ancient-bush-35483136`) to **Launch** before loading facts. Expected ~$0.35/GB-month + compute; demo likely **<$5/month**.

Until then, `core.*` exists empty (except schema), which is enough to publish the dashboard.
