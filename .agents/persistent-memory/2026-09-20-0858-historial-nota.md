# 2026-09-20 08:58 — get_company historial / nota_confianza

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:13 +02
Branch: `agent/overnight-optimize` (tick 87)

## What changed

`get_company` history is `historial`, note is `nota_confianza`, categories are `categorias`. Plot kinds unchanged (`score_history`).

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Still **6 / 6 / 5 / 6 / 6**. `historial.month` is now `julio 2026`, not `jul 2026`. why-score Helmcode stall ~95 s (`--ignore-latency`).

## Decisions

Keep. Do not merge.

## Still unknown

Helmcode stalls.
