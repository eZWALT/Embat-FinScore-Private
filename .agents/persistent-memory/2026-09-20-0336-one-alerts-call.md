# 2026-09-20 03:36 — one get_alerts call; no “Llamo a…”

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 03:36 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `get_alerts` cap is 1. The server unions `entity_id` with the session company/group so one call covers both.
- Empty-input `get_alerts` is the intended path when SESSION already has empresa + grupo.
- BREVITY / WORDING: do not announce calls; do not recite the method at the end.
- Alert `entity` is «Empresa 0016» / «Grupo 0070»; evidence strings rewrite `COMP_*` / `GROUP_*`.

## Decisions

- Fair bench keep. alerts **1 tool / q 6 / 7.8 s** (main 2 / 11 s). period-4 still 4 / q 6. No «Llamo a…» this run.

## Still unknown

- Alerts still sometimes close with lift/stats even when nobody asked for fiabilidad.
