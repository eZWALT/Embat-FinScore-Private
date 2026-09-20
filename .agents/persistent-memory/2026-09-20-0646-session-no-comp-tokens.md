# 2026-09-20 06:46 — SESSION series speak Empresa, not COMP_

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 06:46 +02

## What changed

`formatDashboardView` no longer repeats `COMP_*` / `GROUP_*` on focus or series (the `session` block still has `company_id=` for tools). Leyenda uses spoken trajectory. Screen blurbs are shorter and say Vigilancia, not Alerts. Remaining English tool errors (control / cluster / forecast / records) are Spanish.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6. Period 807 chars, refuse named Empresa 0030.
- Do not merge to main tonight.

## Still unknown

Live Rápido still has the Next overlay on the FAB (pre-existing).
