# Vercel public output directory failure

- **Author:** Codex
- **Timestamp:** 2026-09-19 14:30 CEST

## What changed

The Git-connected Vercel project `hack-spain` (team `hack-spain-3d7d523b`, root `product/web`) had **no framework preset**. Next.js built successfully, then the platform looked for a static `public` output directory and failed.

Set the project framework to **Next.js** and cleared output-directory override. Redeploy of `dpl_4kWBys61du65PfB2YXSPHgHwGobJ` is **Ready**; production alias `https://hack-spain.vercel.app`.

Added `product/web/vercel.json` with `"framework": "nextjs"` so later imports do not fall back to Other/`public`.

A spare personal-account project `embat-health-sentinel` was created while searching; it is unused for this Git deploy.

## Decision

For this App Router app, Framework Preset must be Next.js. Do not set Output Directory to `public`.

## Still unknown

Whether the extra `embat-health-sentinel` project should be deleted.
