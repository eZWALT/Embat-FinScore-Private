# 2026-09-20 08:32 — leftover evidence decimals; chart list in Spanish

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:32 +02
Branch: `agent/overnight-optimize` (tick 93)

## What changed

Alert evidence leftovers that are not money/pct/score speak as a Spanish tenth. Missing control-chart kinds return `disponibles` with Spanish `comparacion`/`metrica`. Month-miss error lists `meses_puntuados`.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Still **6 / 6 / 5 / 6 / 6**. period-4 Helmcode stall ~99 s. Alerts: «sin el tope sería 58», Tesorero action as-is.

## Decisions

Keep. Do not merge.

## Still unknown

Helmcode stalls. Max 08:55 +02.
