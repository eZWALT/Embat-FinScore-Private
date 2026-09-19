# 2026-09-19 1345 — Neon end-to-end load (staging)

- **Author:** Cursor Grok
- **When:** 2026-09-19 13:45 CEST

## What changed

- Created Neon project `embat-health-sentinel-staging` (`sparkling-glitter-12146358`) in `aws-eu-central-1`, default branch `main` left empty, data on branch `staging`.
- Added `infra/neon/` migrations, export/transform/load/validate scripts, HTTP loader (TCP 5432 is blocked here), setup README, validation report.
- Next.js: `NeonScoreRepository` behind `createScoreRepository()`; local bundle remains the fallback. `product/web/.env.example` documents `DATABASE_URL` (server-only).

## Decisions

- Four schemas: `core` (clean DuckDB), `analytics` (immutable score run), `api` (views). Unified `core.products` with `family` in {banking, debt, unknown}. Counterparties synthesized; no FK to `companies`.
- Analytics loaded in full and matches the bundle: 1,286 / 250 / 19,658. Database size 83 MB on Free.
- Do not load ZIPs. Do not enable Data API. App uses `@neondatabase/serverless` over HTTPS.
- Core facts deferred: Free 0.5 GB cap + local 5432 timeout. Launch upgrade needs an explicit yes.

## Still unknown / pending the user

- Confirm Launch (region already Frankfurt) to load transactions/invoices.
- Network path for `psql`/`\\copy` (or run `load.sh` from a host that can reach :5432).
- Vercel `DATABASE_URL` for the hosted app; keep using branch `staging` until they promote.
- `neon login` in the CLI if they want the CLI independently of MCP (MCP already authenticated).
