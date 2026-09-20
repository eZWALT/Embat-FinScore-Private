# 2026-09-20 07:45 — Follow-up owner chips only after get_alerts

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 07:45 +02

## What changed

`sanitizeFollowups` drops dueño/acción chips unless `get_alerts` already ran. Fallback second chip is «¿Qué hay detrás de esos euros?» when there is no alert. `explain_change` speaks `desde` / `hasta` / `cambio`. Records intro and Sentinel refuse match SCOPE (no named off-topic).

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6.
- Live local after «¿Qué explica este gráfico?»: chips «¿Qué hundió las entradas de Empresa 0176?» / «¿Por qué sigue estable Empresa 0030?» — no «quién debe actuar».
- Do not merge to main tonight.

## Still unknown

Opening chips on a multi-company chart still include «¿Qué alertas hay?» (invitation, not a follow-up).
