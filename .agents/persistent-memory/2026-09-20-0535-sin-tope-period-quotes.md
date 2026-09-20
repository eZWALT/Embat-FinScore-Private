# 2026-09-20 05:35 — Explicit sin tope + period quote labels

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 05:35 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `speakGuard` always returns a label: tope or **sin tope**. Period lead can count.
- Period prompts: do not hang cobros on a stable company; do not say «tres… hundidas» if only two payloads have a tope.
- Multi-company tool quotes prefix **Empresa N ·** so ×4 rows are readable.
- Bench: invented dueño ignores «cobros vencidos»; period-4 «tres de las cuatro» + hundidas is −2.

## Watch

Production Pregunta (main, no company): still asks for `COMP_xxxx` / `GROUP_xxxx` after «¿Qué alertas hay?». This branch refuses unscoped alerts and never asks the user for a token.

## Bench

keep=true. Period lead: two fading topes + 0651 mora + 0030 estable sin tope.

## Still unknown

- Do not merge. Loop until 05:55–08:55 +02.
