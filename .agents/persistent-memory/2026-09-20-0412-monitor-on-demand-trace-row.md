# 2026-09-20 04:12 — monitor product layer on demand; tool row shows the summary

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 04:12 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `product_monitor.md` (charts, alert kinds, lift stats, TellMe) loads only for Sentinel or when the question is alertas / gráfico / clúster / previsión. Score facts stay in `product_context.md`.
- `AgentTrace` puts the output summary on the row («Leer índice · Empresa 0030 · índice 88»). No chevron unless there is SQL or an error.

## Decisions

- Fair bench keep. why-score / period-4 no longer ingest the lift lecture. alerts still 1 / q 6 with owner + action.

## Still unknown

- Whether dropping monitor from period-4 ever hides a group-alert fact the user wanted without asking.
