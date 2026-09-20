# 2026-09-20 08:22 — category `pts` spoken as Spanish tenths

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:22 +02
Branch: `agent/overnight-optimize` (tick 81)

## What changed

`slimCategories.pts` uses `speakScore` (`40,5`) instead of a raw float.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Still **6 / 6 / 5 / 6 / 6**. period-4 Helmcode stall ~100 s (`--ignore-latency`).

## Live

Chart chip: Leer índice ×3, quotes with space, lead by guard, no glide, no confidence coda, follow-ups without dueño.

## Decisions

Keep. Do not merge.

## Still unknown

Helmcode stalls. Production Pregunta on main still asks for a company when none is selected (expected).
