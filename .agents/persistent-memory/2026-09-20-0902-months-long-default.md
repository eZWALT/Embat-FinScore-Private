# 2026-09-20 09:02 — tool months default to julio 2026

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:16 +02
Branch: `agent/overnight-optimize` (tick 88)

## What changed

`speakMonth` defaults to the long label. Control-chart, forecast points, group history, and error `scored_months` no longer say `jul 2026`.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Still **6 / 6 / 5 / 6 / 6**.

## Decisions

Keep. Do not merge.

## Still unknown

Helmcode stalls. Max 08:55 +02.
