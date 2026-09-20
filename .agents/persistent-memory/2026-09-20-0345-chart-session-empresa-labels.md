# 2026-09-20 03:45 — Empresa labels on the chart and in SESSION

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 03:45 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- Rápido chart legend + compare chips use `companyLabel` («Empresa 0030»), not `COMP_0030`.
- SESSION series lines: `azul: Empresa 0030 (COMP_0030) Grupo 0126 … estable`. Trajectory in Spanish. The raw id stays in parens so tools still get it.

## Decisions

- Local Pregunta on Empresa 0030: «Leer índice» + two bundle `sentence` quotes, then the answer. No raw JSON.
- Production hack-spain Pregunta is still main.

## Still unknown

- Next hydration overlay can still cover the FAB on localhost.
