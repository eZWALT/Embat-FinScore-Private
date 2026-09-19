# 2026-09-19 1450 — AGENTS.md storage is Neon

- **Author:** Cursor Grok
- **When:** 2026-09-19 14:50 CEST

## What changed

Updated root `AGENTS.md` so product storage matches production: Next.js on Vercel reads Neon Postgres. Pointers now include `infra/neon/`. Step 5 no longer says pick a stack / empty Dockerfile. The storage section treats the export bundle and clean DuckDB as immutable **sources** that load into `analytics` / `core`; the app reads `api`. Live sizes, project IDs, and which facts are loaded stay in the journal and `infra/neon/README.md`.

## Decisions

- Do not put Neon/Vercel live status in `AGENTS.md`.
- Do not serve the JSON bundle or DuckDB from the hosted app.

## Still unknown

- Whether `core` facts (transactions/invoices) get loaded after a Launch upgrade.
