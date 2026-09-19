# 2026-09-19-1845 — Neon `neondb` reloaded with the schema 1.3.0 bundle (gliding guard)

- **Author:** Claude Code (Sonnet 5) for Javier Boix
- **Target:** the same Neon database as `2026-09-19-1700-neon-fresh-db-load.md` (pooled endpoint `ep-aged-block-b2bi6pi0`, Postgres 17), at Javier's request. The connection string was passed only as the `DATABASE_URL` environment variable of each command; it is not stored in any file of the repo. It was pasted in a chat again: **rotate the `neondb_owner` password**.

## What was done

1. Bundle: `~/Desktop/Embat-handoff/bundle` (schema 1.3.0, scorecard v1.1, 1,286 companies, 250 groups, 2,374 alerts), from commit `672cd09`.
2. `PYTHONUTF8=1 python infra/neon/scripts/transform_bundle.py --bundle <bundle> --run-id 31490554-3a48-5fdf-baa7-27eac8ea5708` (uuid5 of URL namespace + `embat-score-bundle-1.3.0-es`).
3. `node infra/neon/scripts/load_http.mjs --analytics-only` (2 min): migrations 001-006, truncate + COPY. The previous run (1.2.0, 85c811c0-...) is gone; `analytics.score_runs` has one row.
4. Verified over HTTPS against the bundle files: one run 1.3.0 / v1.1; `api.current_run` 1,286 / 250 / 19,658; 2,374 alerts; no `nan` / `-0 %` in alert text; **every one of the 19,658 company-month scores, `score_pre_cap` and `guard` equals the bundle**; forecast fans ordered inside 0-100; month-on-month falls of 20+ points 48, of 30+ 2 (as in the bundle). Database size 86 MB.

## Not done / still open

- `guard_ceiling` and `spec.guard.max_drop_per_month` (new in 1.3.0) are **not in Postgres**: no columns and the transform does not read them. `analytics.score_runs.monitor`/`spec` jsonb, if it carries the spec, has the new field; the per-month ceiling would need a column and a transform change.
- The forecast fields listed as not loaded in the 1700 note are still not loaded. `core` is still empty (Free plan cap).
- The web app was not started against this database in this session.
