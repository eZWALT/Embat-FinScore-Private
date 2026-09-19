# Vercel web build fix

- **Author:** Codex
- **Timestamp:** 2026-09-19 12:43 CEST

## What changed

- Anchored the root Python `lib/` ignore rule as `/lib/`.
- Added the previously ignored `product/web/src/lib/` application modules to version control.
- Verified `pnpm lint` and the production `pnpm build` from `product/web`.

## Decision

Application source directories must not be covered by broad Python-oriented ignore rules. Vercel continues to use `product/web` as the project root.

## Still unknown

- The next Vercel deployment must confirm that the checked-out monorepo exposes `product/score/sample_bundle` during the static build.
