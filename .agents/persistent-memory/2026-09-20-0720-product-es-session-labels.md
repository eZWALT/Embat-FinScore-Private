# 2026-09-20 07:20 — PRODUCT in Spanish; SESSION without COMP_*

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 07:20 +02

## What changed

Always-on PRODUCT is Spanish (same weights, tope, trajectory, 39 % sin facturas, no hidden-test). SESSION extra speaks `empresa=` / `grupo=` labels, not `company_id=COMP_*`.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6. Refuse still the 85-char plantilla. why-change 425 chars cites 81,1 and 535 k€.
- Leftover: period bullet still writes `−535 k€` (WORDING 7: `eur` is the amount).
- Do not merge to main tonight.

## Still unknown

Whether a signed-`eur` bench extra would drop period quality too often.
