# 2026-09-20 08:08 — reason `points` spoken as signed Spanish

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:08 +02
Branch: `agent/overnight-optimize` (tick 79)

## What changed

`slimReason.points` is now `speakDelta` (`−8,6`), same tenth as the index. The model no longer sees raw `-8.55`.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Full suite **6 / 6 / 5 / 6 / 6**. Period cites `−8,5 pts, 535 k€` (sign on points, amount unsigned). why-change cites 72,6 / 81,1 / 535 k€ / −8,6. No extras.

## Decisions

Keep. Do not merge.

## Still unknown

Helmcode stalls. Live Pregunta quotes after clip still to confirm on the three-company chart.
