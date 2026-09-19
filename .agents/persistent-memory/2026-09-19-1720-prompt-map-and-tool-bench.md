# 2026-09-19-1720 — Prompt map (full stack kept) + tool-call bench

- **Author:** agent (Cursor / Walter)
- **When:** 2026-09-19 17:20 CEST

## What changed

- Full Ask prompt stays complete. Context is not trimmed.
- New first layer: `product/web/src/lib/agent/prompts/prompt_map.md` (MAP). Loader order is now MAP → ROLE → SCOPE → PRODUCT → WORDING → FORMAT → TOOLS → RECORDS → SESSION.
- Each prompt file is titled as its layer so a human can see which job it has.
- Conflict order (stated in the map): SCOPE > WORDING > TOOLS > PRODUCT > RECORDS.
- `scripts/bench-ask.mjs` now matches that loader and measures Helmcode **time-to-first-tool-call** (auto × 3 + required × 1) with the ten Ask tools.

## Decisions

- Do not shrink `product_context.md` or `tools_catalog.md` to chase TTFT.
- MAP is an index, not a second copy of the product facts.

## Still unknown

- Neon execute ms (no local `DATABASE_URL`).
- Numbers land in `product/web/src/lib/agent/BENCHMARK.md` after `node scripts/bench-ask.mjs`.
