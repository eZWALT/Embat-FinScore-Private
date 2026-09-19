# 2026-09-19-1525 — Watcher opens with a set; sparks + tool plots

- **Author:** Walter (agent)
- **When:** 2026-09-19 ~15:25 CEST

## What changed

`/watcher` with no query redirects to the largest group so the channel is not an empty picker. Each month post can carry a 6-month sparkline of the focus company (deterministic, not LLM). Ask / Watcher replies render `plot_series` output as a Recharts line.

## Decisions

Still no LLM on first paint. Spark is part of the renderer, not a new prose field.

## Still unknown

Live Helmcode on Vercel (`HELMCODE_API_KEY`). Neon `core` for record questions.
