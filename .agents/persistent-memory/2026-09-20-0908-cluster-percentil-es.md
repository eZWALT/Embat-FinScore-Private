# 2026-09-20 09:08 — cluster vs_pares numbers spoken

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:19 +02
Branch: `agent/overnight-optimize` (tick 89)

## What changed

`compare_with_cluster` `percentil` and `z` use `speakScore`.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Still **6 / 6 / 5 / 6 / 6**. Refuse stall ~92 s, plantilla unchanged. why-change cites 82,6 julio → 72,6 agosto.

## Decisions

Keep. Do not merge.

## Still unknown

Helmcode stalls. Max 08:55 +02.
