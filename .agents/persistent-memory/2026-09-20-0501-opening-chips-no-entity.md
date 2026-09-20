# 2026-09-20 05:01 — opening chips without a company

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 05:01 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- Opening chips on Rápido without empresa/grupo/series are «¿Qué empresa miro primero?» / «¿Quién está peor este mes?», not «¿Qué alertas hay?».
- `get_group` tool description no longer promises `alert_ids`.

## Decisions

- Production Pregunta on Rápido (no company) still offers «¿Qué alertas hay?» and then asks for a `COMP_*`. This branch does not offer that chip until there is an entity.
- Fair sample keep (same tools / q).

## Still unknown

- `product_context.md` is still English and still names `dark` / `fading`.
