# 2026-09-20 08:30 — item keys in Spanish; alert row plural

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:04 +02 (journal slug keeps overnight series)
Branch: `agent/overnight-optimize` (tick 82)

## What changed

Fallback `items` keys use `REASON_LABELS` (`La volatilidad de salidas`, not `out_vol`). Alert tool row: `1 alerta · 1 entidad`.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Still **6 / 6 / 5 / 6 / 6**. why-change 442 chars still cites 72,6 / 81,1 / 31 k€ / 847 €.

## Live

Empresa 0016 alerts: one `get_alerts`, Tesorero / CFO quotes, actions as-is, follow-up «¿Qué debe hacer el Tesorero?» only after alerts.

## Decisions

Keep. Do not merge.

## Still unknown

Helmcode stalls. `1 entidades` was UI-only; HMR will show the plural fix.
