# 2026-09-20 07:52 — explain_change deltas use Spanish item labels

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 07:52 +02

## What changed

`explain_change` `deltas` keys go through `REASON_LABELS` (`La volatilidad de salidas`, not `out_vol`). `REASON_LABELS` is exported from `plain-language.ts`.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6. why-score Helmcode stall ~97 s.
- Production unscoped «¿Qué alertas hay?» still asks the user for `COMP_xxxx` (this branch does not).
- Leftover: why-change once rounded 72,6 → 73.
- Do not merge to main tonight.

## Still unknown

`explain_change` is rare now (`get_company` already has `change_reasons`).
