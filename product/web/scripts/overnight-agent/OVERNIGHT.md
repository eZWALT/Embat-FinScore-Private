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
