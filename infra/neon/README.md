# Neon (Health Sentinel)

Postgres for the Next.js app: cleaned source (`core`), score-run outputs (`analytics`), and read models (`api`). The browser never receives `DATABASE_URL`.

## What is loaded today

Staging project `embat-health-sentinel-staging` (`sparkling-glitter-12146358`), region `aws-eu-central-1` (Frankfurt), branch `staging`.

- **Analytics (done):** full score bundle, 1,286 companies / 250 groups / 19,658 company-months, ~83 MB. Fits the Free 0.5 GB cap.
- **Core facts (not loaded):** `clean.transactions` (2,556,068) and `clean.invoices` (896,711). TCP 5432 from this network times out; HTTP insert of 2.5M rows would also blow past Free storage once indexes exist.

ZIPs (`bundle.zip`, `embat_clean.duckdb.zip`) are duplicates and are never loaded.

## Setup

1. Official CLI: `npm i -g neon` then `neon login` (browser OAuth). MCP can also create projects.
2. Unpooled connection string: `neon connection-string staging --project-id sparkling-glitter-12146358`
3. Copy `product/web/.env.example` to `product/web/.env.local`. Put `DATABASE_URL` there and in `infra/neon/.env`. Never commit either file.
4. Next.js uses `@neondatabase/serverless` over HTTPS. Local (`.env.local`) and Vercel (`DATABASE_URL` server env) both query Neon on each request. There is no bundle fallback.

## Migrate

```bash
python3 infra/neon/scripts/audit_sources.py
python3 infra/neon/scripts/transform_bundle.py
# optional DuckDB export (includes large fact CSVs):
python3 infra/neon/scripts/export_duckdb.py

# Preferred when port 5432 works:
DATABASE_URL='postgresql://…' infra/neon/scripts/load.sh --analytics-only
DATABASE_URL='postgresql://…' infra/neon/scripts/load.sh --core-only

# This network: HTTP loader (analytics):
node infra/neon/scripts/load_http.mjs --analytics-only
```

Loads are idempotent: they `TRUNCATE … CASCADE` the target schema then COPY/insert.

Apply SQL in order: `001` schemas → `002` core tables → `003` analytics tables → COPY → `003b` map children → `004` indexes/FKs → `005` views → `006` security.

## Validate

```bash
psql "$DATABASE_URL" -f infra/neon/scripts/validate.sql
psql "$DATABASE_URL" -f infra/neon/scripts/explain.sql
```

Report: `infra/neon/reports/validation.md`.

## Rollback

- **Data:** re-run the loader (truncate + reload) or `TRUNCATE analytics.score_runs CASCADE`.
- **Branch:** keep `main` empty; restore `staging` from parent if needed (`neon branches reset staging` only with explicit approval).
- **Project:** do not delete the Neon project without approval.

## Refresh after a new CSV drop

1. Rebuild the export bundle and clean DuckDB (`python -m product.score.export`, `python -m product.score.clean_db`).
2. Re-run transform + load. New `run_id` replaces the previous analytics run.
3. Point the web app at the same `DATABASE_URL`; views always read `api.current_run`.

## Cost

Free plan: $0, 0.5 GB/project, 100 CU-hours. Analytics-only is ~83 MB.

Full core + indexes is expected in the 0.6–1.0 GB range → **Launch** (pay as you go): storage **$0.35/GB-month**, compute **$0.106/CU-hour**, scale-to-zero after 5 minutes. Demo traffic is typically well under $5/month; confirm before upgrading.
