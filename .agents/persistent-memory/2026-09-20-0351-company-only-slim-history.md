# 2026-09-20 03:51 — company-only force text; slim history unless period

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 03:51 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- After one successful `get_company` on a single-company question (no alertas / periodo / registros / gráfico), strip tools and write.
- `score_history` is 4 months unless the question is a dragged period (then 18).
- `get_company` no longer ships `alert_ids`; the feed is `get_alerts`.

## Decisions

- Fair bench keep. period-4 still 4 / q 6 (18-month path). why-score 1 / q 6 / 6.2 s.

## Still unknown

- Whether a 4-month history ever hides a reason the user needed for “este mes”.
