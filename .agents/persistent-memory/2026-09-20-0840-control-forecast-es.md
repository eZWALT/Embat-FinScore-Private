# 2026-09-20 08:40 — Control-chart and forecast payloads in Spanish

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 08:40 +02

## What changed

`get_control_chart` speaks `comparacion` / `metrica` / `persistencia` / `metodo`. `get_forecast` `metodo` is «último valor» instead of `naive_last`. Input enums stay English so retries work.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6. Period lead clean; why-change cites 72,6 / 81,1.
- Do not merge to main tonight.

## Still unknown

Control/forecast are not in the numeric suite.
