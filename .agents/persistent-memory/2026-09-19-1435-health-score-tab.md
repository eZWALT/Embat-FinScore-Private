# 2026-09-19 1435 — Health Score sidebar tab

- **Author:** Cursor Grok
- **When:** 2026-09-19 14:35 CEST

## What changed

- Sidebar has a **Health Score** view: only the 0–100 time series, with a searchable picker of all 1,286 companies (up to 8 overlayed).
- Resumen keeps the existing company detail. Clicking Evolución / Categorías / Señales returns to that view.

## Decisions

- Cap at 8 series so the chart stays readable; search to reach any `COMP_*`.
- Charts mount client-only to avoid Recharts hydration mismatches.

## Still unknown

- Whether the cap of 8 should be raised, or a ranking table added later.
