# 2026-09-20 05:32 — Empresa labels + period history span

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 05:32 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `get_company` / `explain_change` / `get_group` speak `empresa` / `grupo` (Empresa 0030), not `COMP_*`. Inputs still accept `COMP_xxxx` or «Empresa 0030».
- Period `score_history` is the dragged span + 1 month (junio→agosto → 4), not 18 months × 4 companies.
- Never write a bare `0651` or «la de 65 puntos».
- Bench flags invented dueño, leaked tokens, and bare ids; compare recomputes quality on both sides.

## Bench (sample vs main, `--ignore-latency`)

keep=true. why-score / alerts / refuse / period-4 / why-change all q 6/6/5/6/6. Period lead names Empresa 0011 / 0176 / 0651 / 0030. No invented dueño.

## Decisions

- Same tools. Do not merge tonight.

## Still unknown

- Period lead can still lump 0030 with cobros. Loop until 05:55–08:55 +02.
