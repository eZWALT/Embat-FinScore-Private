Author: Codex
Timestamp: 2026-09-19 13:03 CEST

## What changed

- Reviewed Neon official documentation for CLI/API provisioning, PostgreSQL imports, serverless Next.js connectivity, and current pricing/limits.
- Confirmed the local machine has Node, pnpm, and `psql`, but not the Neon CLI or DuckDB CLI; no Neon credentials are exposed in the current environment.

## Decisions

- Neon can be handled end to end from this repo through the official CLI/API after a one-time user OAuth login (`neon auth` or `neon init`).
- Use ordinary Neon Postgres plus a server-only Next.js repository adapter; do not expose database credentials to the browser.
- Export the supplied DuckDB and computed JSON bundle into bulk-loadable files, create schema first, load data before secondary indexes/foreign keys, then validate row counts, null/orphan checks, and database size.
- The unclaimed/claimable Neon flow is unsuitable because its temporary project limit is below the supplied logical payload.

## Still unknown

- Neon account/organization, desired EU region, and whether billing should be enabled.
- Final choice between loading the complete clean DuckDB plus computed bundle or only product-serving tables.
- Vercel project access and production environment-variable authorization for the final deployment step.
