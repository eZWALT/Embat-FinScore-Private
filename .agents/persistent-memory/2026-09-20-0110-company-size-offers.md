# Company size for the commercial offers

Author: Javier Boix (with Claude). 2026-09-20 01:10.

## What changed

- `product/web/src/config/offers.ts`: size bands (`MAGNITUDES`, by mean monthly operating inflow: Micro <10 k, Pequeña 10–100 k, Mediana 100 k–1 M, Grande 1–10 M, Muy grande >10 M), a minimum size per product (`size`) and an indicative amount per product (`ticket`, a multiple of inflow, outflow or cash). All the numbers are placeholders to be reviewed by someone who knows the business.
- `product/web/src/lib/offers.ts`: `guidanceFor(company, size)` drops products too big for the company and attaches an amount range.
- `product/web/src/components/offers/offer-guidance.tsx`: the Sugerencia comercial card always lists the bands, highlights the company's, and has a Prudente / Ambicioso toggle for the amount. It says so when size is unavailable.
- `infra/neon/migrations/007_company_size.sql`: new table `analytics.company_size` (one row per company per run, `ON DELETE CASCADE` from `score_runs`, RLS on like the others).
- `infra/neon/scripts/export_company_size.py`: writes the SQL that fills it from the clean DuckDB. No connection to Neon; run the output in the Neon SQL editor (or any client).
- `product/web/src/lib/data/size.ts` and `/api/size`: read the stored row of the current run, fall back to `core.*` when there is none.

## Loaded

Applied 007 and loaded 1,286 rows into the staging Neon project on 2026-09-20 (window Jun–Aug 2026, as-of 2026-08). 173 companies have no inflow in the window; 13 have no cash snapshot.

## Decisions

- `core.*` is empty in Neon (0 rows in companies, transactions, balances), so size cannot be computed at request time. Loading `core.transactions` (2.5M rows) does not fit the free tier; a 1,286-row table does.
- Size = mean monthly operating inflow over 3 months, categories as `CAT_MAP` in `analysis/features/common.py`. Checked against the feature store (`a_op_in`, `a_op_out`) for 1,224 companies: max difference 2e-6.
- Cash = sum of positive account balances on the latest snapshot. `countable` is filled for only 296 companies, so it is not used.
- Amounts are in the company's currency, not converted. The bands read as euros.

## Still unknown

- Whether the band edges, minimum sizes and ticket multiples match how Embat sizes offers.
- After a new bundle load the table is empty for the new run (the cascade drops it). Re-run `export_company_size.py` and paste the output.
- The Rápido suggestion strip does not use size yet.
