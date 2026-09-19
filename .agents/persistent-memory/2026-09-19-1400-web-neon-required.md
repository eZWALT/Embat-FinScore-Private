# 2026-09-19 1400 — Next.js always reads Neon

- **Author:** Cursor Grok
- **When:** 2026-09-19 14:00 CEST

## What changed

- `createScoreRepository()` requires `DATABASE_URL` (no silent sample-bundle fallback).
- Home page is `force-dynamic` so local `pnpm dev` and Vercel production query Neon per request instead of prerendering the dashboard.

## Still unknown

- This repo is not a Vercel project yet (only `trailrunningcal` on the team). Production needs a linked project plus server-only `DATABASE_URL`.
