# 2026-09-19 1530 — Production UI up, data load fails

- **Author:** Cursor Grok
- **When:** 2026-09-19 15:30 CEST

## What changed

Diagnosed https://hack-spain.vercel.app/: Next.js deploy is Ready (`dpl_FCJU8vBLY6bBd13A2Rf9NVBPyKFT`, HTTP 200). Home Server Component throws (RSC digest `2307198119`); error UI still says “bundle local”.

Vercel root is `product/web`, so `product/score/sample_bundle` is not on the filesystem. App now requires server-only `DATABASE_URL` → Neon. Staging branch `br-winter-cake-b2m43sob` has `api.manifest` (1,286 companies); default `main` does not.

MCP Vercel token is the personal team (`trailrunningcal` only), not `hack-spain-3d7d523b`, so env vars on the live project could not be listed.

## Still unknown

Whether `DATABASE_URL` is set on the hack-spain Vercel project, and whether it points at staging vs empty `main`.
