# 2026-09-20 08:15 — Spanish SQL-guard errors; plot titles say Empresa

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 08:15 +02

## What changed

`sql-guard` / `query_clean_db` errors are Spanish (`rechazada`, `solo SELECT`). Plot titles, series keys and axis labels use `entityLabel` (Empresa/Grupo), not `COMP_*`.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6 (why-change hit `rounded_score` once). Plots are not in the suite.
- Do not merge to main tonight.

## Still unknown

A live `plot_series` on local Pregunta was not re-clicked this tick.
