# 2026-09-20 06:52 — Peers questions are not company-only

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 06:52 +02

## What changed

`companyOnly` (force text after one `get_company`) is false when `wantsGroupTools` is true, so «grupo de pares» / «este grupo» can still call `get_group` / `compare_with_cluster`. Why-score stays one `get_company`.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6.
- Do not merge to main tonight.

## Still unknown

Local period on Empresa 0011 / 0176 / 0030: Leer índice ×3, one quote per empresa, lead by guard, chips without `COMP_*`. One extra summary sentence after the bullets (variance).
