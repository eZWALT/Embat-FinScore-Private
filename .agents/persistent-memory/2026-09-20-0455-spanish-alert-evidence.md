# 2026-09-20 04:55 — Spanish alert evidence

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 04:55 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `get_alerts` drops `alert_id`. Persistence is `N de los últimos 4 meses`.
- Evidence keys/values in Spanish: `índice`, `habitual`, `sin_tope`, `miembros` as Empresa, money as `14 k€`, share as `3 %`, category as «Liquidez y deuda». Nulls dropped.

## Decisions

- Fair sample keep. Alerts 1305 chars (cites Javi action quotes) still q 6 / 1 tool. why-score tightened to 606 chars.

| Case | main tools / q | tick 16 tools / q / s |
|---|---|---|
| why-score | 1 / 6 | 1 / 6 / 5.4 |
| alerts | 2 / 6 | 1 / 6 / 8.8 |
| refuse | 0 / 5 | 0 / 5 / 2.2 |
| period-4 | 11 / 2 | 4 / 6 / 12.8 |
| why-change | 2 / 6 | 1 / 6 / 8.3 |

## Still unknown

- `summary` + `title` + `action` still overlap; next candidate can drop `summary` if quality holds.
