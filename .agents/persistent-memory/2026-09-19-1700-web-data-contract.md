# 2026-09-19-1700 — data contract and static bundle for the web app

- **Author:** Claude Code (Sonnet 5) for Javier Boix
- **Why:** colleagues start building the product (Next.js on Vercel) while the pipeline runs once and ships results. They need the schema and a way to feed it, and the schema must already carry what steps 3-5 will add.

## What exists

- `python -m product.score.export --csv-folder <dir> --out <bundle>`: pipeline -> score -> static JSON bundle, ~45 s. Files: `manifest.json` (version, months, section status, spec with labels/weights/units, disclaimer, sha256), `companies.json` (index + sparklines), `companies/{id}.json` (all scored months), `groups.json`, `types.ts`. Full bundle: 1,286 companies, 41.6 MB raw, ~4 MB gzipped at `--detail-months 12` (items, reasons, attribution only for the last 12 months; every month keeps score, categories, trajectory, confidence).
- `product/score/DATA_CONTRACT.md` (semantics, layout, rules, how the app grows, relation to the v0 dummy/POC), `contract/types.ts` (compiles under `tsc --strict`), `contract/loader.example.ts` (disk at build time or `BUNDLE_URL`), `sample_bundle/` (12 varied companies, 0.4 MB, committed).
- `export --validate <dir>`: counts, score range, enums, contributions sum to score (±0.15), attribution sums, ≤4 reasons, sha256. Passes on the sample and the full bundle; tampering with a score is caught.
- `run.score_folder` / `score_store(return_items=True)` added so export reuses the score path.

## Forward compatibility (plan steps 3-5)

Schema 1.0.0 already defines, but does not yet emit: `ControlChart` (four comparisons, EWMA/CUSUM, persistence flag), `ClusterIndex`/`ClusterMembership`, `AlertFeed`/`Alert` (two-sided direction, severity, reasons with €, persistence, owner, action, evidence, rank score; top-customer-quiet wording constraint), `Forecast` (fan with naive baseline), `EntityType` incl. customer/supplier (blocked: counterparty IDs do not map to company IDs). `manifest.sections` marks each available/planned/blocked so the UI can hide or stub. Additive fields = minor version, consumers ignore unknown fields.

## Decisions

- Static files, not an API: the run is one-off. Full bundle not committed; deploy it with the build or host it (Blob/S3) via `BUNDLE_URL`. Vercel plan limits not checked by me.
- Reasons stay English sentences plus structured `item/value/unit/eur/points` so the UI can template Spanish. The v0 card has `_es`; this does not.
- Only the v1 scorecard is exported. The POC still reads the v0 parquet; mapping in DATA_CONTRACT.md. Which card is the product's is undecided.

## Still unknown

- Whether the app needs a group-level detail file or search beyond `companies.json`.
- The alert/control-chart/forecast shapes are my proposal from the plan text; step 3 may change them (minor bump if additive).
- Nothing committed yet.
