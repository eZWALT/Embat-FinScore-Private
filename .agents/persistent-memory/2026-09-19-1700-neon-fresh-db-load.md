# 2026-09-19-1700 — Neon: analytics loaded into a fresh database (schema 1.2.0, Spanish bundle)

- **Author:** Claude Code (Sonnet 5) for Javier Boix
- **Target:** Neon database `neondb`, pooled endpoint `ep-aged-block-b2bi6pi0`, Postgres 17. It was **empty** when I connected (only `public`, 7 MB). It is **not** the populated staging branch in `2026-09-19-1345-neon-staging-load.md` (project `sparkling-glitter-12146358`, branch `staging`): check which one the Vercel `DATABASE_URL` points to.
- Credentials: the connection string is only in the gitignored `infra/neon/.env` on this machine. It was pasted in a chat, so **rotate the `neondb_owner` password**. The S3 storage keys pasted earlier are not used by anything in the repo and should be revoked too.

## What was done

1. Bundle: full export from the current `main` code (Spanish text, schema 1.2.0, 1,286 companies, 250 groups, 2,365 alerts).
2. `infra/neon/scripts/transform_bundle.py --bundle <bundle> --run-id 85c811c0-778f-5767-8f12-345fb38ea587` (uuid5 of `embat-score-bundle-1.2.0-es`; run with `PYTHONUTF8=1`, the script reads files with the default encoding and the bundle is now UTF-8 Spanish).
3. `node infra/neon/scripts/load_http.mjs --analytics-only`: migrations 001-006, then the analytics COPYs. Only the `@neondatabase/serverless` package was installed into `product/web/node_modules` (untracked, `--no-save`; `pnpm` is not installed on this machine).
4. Verified over HTTPS against the bundle files (not just the loader's own counts): 1,286 / 250 / 19,658, 2,365 alerts, 1,282 forecasts / 7,692 points, fans ordered inside 0-100, no orphans, no `nan`/`-0 %`/English in alert text, Spanish cluster labels, and 13 sampled companies identical to the bundle (score, months, forecast fan, alert title/summary/action). Database size 83 MB.

## Not done / still open

- The new forecast fields (`drivers`, `own_average`, per-point `method`, `skill_by_horizon`, `manifest.monitor.forecast`) are **not in Postgres**: the tables have no columns for them and the transform does not read them. The stored `forecasts.skill_vs_naive` has the new meaning (pinball skill, about +11.5% at 3 months). Needs a migration `007` and a transform change; `analytics.score_runs.monitor` (jsonb) already carries the manifest block.
- `core` (transactions, invoices, ...) is empty here (Free plan cap). The Ask tools that query record tables have nothing to read.
- The web app was not started against this database in this session.
- `validate.sql` needs `psql`; not run. Equivalent checks were run with a Node script over HTTPS.
