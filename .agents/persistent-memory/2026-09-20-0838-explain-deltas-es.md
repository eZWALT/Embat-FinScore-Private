# 2026-09-20 08:38 — explain_change deltas spoken

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:08 +02
Branch: `agent/overnight-optimize` (tick 83)

## What changed

`explain_change` `deltas` and `cambio_tope` use `speakDelta` (`−0,4`). List-companies blurb says `grupo`, not `group_id`.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Still **6 / 6 / 5 / 6 / 6**. Suite still uses `get_company` not `explain_change`; the payload is ready when change_reasons are missing.

## Decisions

Keep. Do not merge.

## Still unknown

Helmcode stalls. Production main still fans out `explain_change` on period-4.
