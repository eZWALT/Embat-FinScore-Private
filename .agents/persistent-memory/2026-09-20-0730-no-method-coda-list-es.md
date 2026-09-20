# 2026-09-20 07:30 — No method coda; list/group speak Empresa

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 07:30 +02

## What changed

WORDING: do not close with «explicable y monitorable»; «sin el tope sería», never «sin el límite». `list_companies` rows are `empresa` / `grupo`. `get_group` members are `empresa`. Bench extras flag `method_coda`.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6. why-change dropped the method coda. Production refuse on hack-spain is already the plantilla with Empresa 0030 (no COMP_*).
- Do not merge to main tonight.

## Still unknown

Production unscoped alerts still ask for `COMP_*` (this branch refuses).
