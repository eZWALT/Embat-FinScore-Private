# 2026-09-20 07:58 — Tool schemas ask for Empresa, not COMP_*

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 07:58 +02

## What changed

Zod describes on company/group ids are `Empresa 0030` / `Grupo 0126`. `asCompanyId` / `asGroupId` still accept `COMP_*`. `list_companies` normalizes `Grupo` before filtering.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6. Period and why-change now call `get_company` with `Empresa 0011` (not `COMP_0011`).
- Leftover: why-change still rounded 72,6 → 73.
- Do not merge to main tonight.

## Still unknown

why-score still passed `COMP_0030` from the bench session id.
