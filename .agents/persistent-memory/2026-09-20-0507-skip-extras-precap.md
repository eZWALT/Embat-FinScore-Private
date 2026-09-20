# 2026-09-20 05:07 — skip Neon extras and unused pre_cap

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 05:07 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `get_company` skips `loadScoreExtras` when `reasons` exist (no Neon extras query on the common path).
- `score_pre_cap` only when a guard is active. `slope3` / `slope6` dropped (trayectoria already speaks).

## Decisions

- Fair sample keep. why-score still 88,5 / 34 k€ / sin tope. why-change still «entradas hundidas (tope 50)» and «sin el tope sería 81,1». period-4 4 / q 6 / 9.4 s.

## Still unknown

- `loadScoreExtras` still runs for `explain_change` (rare after the cap).
