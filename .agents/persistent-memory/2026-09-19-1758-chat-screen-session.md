# 2026-09-19-1758 — Pregunta knows the open dashboard

- **Author:** agent (Cursor / Walter)
- **When:** 2026-09-19 17:58 CEST

## What changed

- The popup gets the open screen in SESSION (mode, chart, color → `COMP_*`, dragged period, Resumen facts). Not a new tool: Neon cannot see the UI, and a retrieve-first hop would add latency.
- Subtitle is one line for every mode: «Sobre los paneles, las tendencias o un periodo.»

## Decisions

- Encode view state in the prompt; keep tools for reasons, €, alerts, history.
- Legend values (current score, Δ, color) may be repeated as what is on screen. Everything else still comes from a tool.
