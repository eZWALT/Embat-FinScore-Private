# 2026-09-20 05:50 — Follow-up chips never COMP_*

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 05:50 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- Follow-up chip prompt: never `COMP_*` or a bare `0651`.
- SESSION `nombra=Empresa 0011 · …` from entities in the question, so a period lead does not shorten to `0011 y 0176`.

## Bench

keep=true. Period q=6, bare_id false, two fading topes + 0651 mora + 0030 estable.

## Still unknown

- Period lead bare-id variance. Do not merge. Min 05:55 / max 08:55 +02.
