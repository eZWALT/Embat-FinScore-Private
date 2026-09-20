# 2026-09-20 06:48 — Period payload drops confidence

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 06:48 +02

## What changed

Period `get_company` no longer ships `confidence` (note and categories already gone). A first slim of PRODUCT/WORDING 2 made the model add «En 0011, 0030 y 0176 la confianza es media» (bare ids, q=5). Dropping the field + «nunca En 0011» restored q=6. Also dropped the 10 pts/month glide sentence from always-on PRODUCT and the “alerta significa…” recitation bait from WORDING 2.

## Decisions

- Keep vs main: 1/6, 1/6, 0/5, 4/6, 1/6. Period 857 chars, no bare ids.
- Do not merge to main tonight.

## Still unknown

why-score still needs confidence; that is intentional.
