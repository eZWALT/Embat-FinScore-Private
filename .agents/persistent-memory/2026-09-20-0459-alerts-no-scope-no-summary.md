# 2026-09-20 04:59 — no unscoped alerts, no summary

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 04:59 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `get_alerts` without empresa/grupo (session, named, or `entity_id`) returns `{ error: "sin empresa", ask: "…" }` — never the feed.
- `slimAlert` drops `summary` (title + action + reasons are enough).
- Catalog no longer promises `alert_ids` / resumen.
- WORDING / ROLE: never ask the user for `COMP_xxxx`.

## Decisions

- Production Pregunta on Rápido (no company) already asked which entity, but leaked `COMP_*`. This branch asks «Empresa 0030».
- Fair sample keep. Alerts 1027 chars (was 1305) / q 6 / 1 tool. Helmcode ~95 s TTFT on that case — ignore latency.

| Case | main tools / q | tick 17 tools / q |
|---|---|---|
| why-score | 1 / 6 | 1 / 6 |
| alerts | 2 / 6 | 1 / 6 |
| refuse | 0 / 5 | 0 / 5 (canonical recusa) |
| period-4 | 11 / 2 | 4 / 6 |
| why-change | 2 / 6 | 1 / 6 |

## Still unknown

- Rápido opening chip «¿Qué alertas hay?» with no company still fires that empty-scope path on production.
