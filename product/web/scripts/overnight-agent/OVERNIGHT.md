# Pregunta overnight log

Branch `agent/overnight-optimize`. Compare every keep to `baseline.json` (main @ `hack-spain.vercel.app`).

## Tick 0 — 2026-09-20 01:55 +02

First pass (before first live candidate bench):

- One retrieval per company: `toolCallKey` ignores month; parallel same-entity calls share one execute.
- `get_company` cap 4 (period chart), still 1 per empresa.
- Catalog + ROLE: no per-month fan-out; no `explain_change` if `change_reasons` is already there; one `get_alerts` with `since_month`.
- Catalog in Spanish (same tools).
- Follow-up chips `line-clamp-3`.
- Leftover Spanish-only SCOPE / WORDING / ROLE from the working tree.

## Tick 0b — baseline (main production)

`hack-spain.vercel.app` · COMP_1186 · 2026-09-19T23:59Z

| Case | tools | q | ms | notes |
|---|---:|---:|---:|---|
| why-score | 1 | 6 | 103s | one `get_company` |
| alerts | 2 | 6 | 8s | company + group |
| refuse | 0 | 5 | 3s | Spanish refuse |
| period-4 | 4 | 5 | 113s | one company + plot; text present |
| why-change | 2 | 5 | 13s | |

Local :3010 is sample_bundle (12 companies). COMP_1186 is not there — do not treat local 1186 benches as vs-main. Preview with Neon is the fair candidate. Local is for compile + sample companies (`COMP_0030`).
