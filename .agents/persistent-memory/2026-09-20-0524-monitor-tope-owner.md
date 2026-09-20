# 2026-09-20 05:24 — Slim monitor + tope per company + owner only on alerts

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 05:24 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- On-demand `product_monitor.md` compressed (same five alerts, same stats, same TellMe).
- Period tope is per-company: only attribute `guard` to payloads that carry it. Lead must not say «tres… tope» when 0651 is cobros.
- Dueño/acción only if `get_alerts` returned them. WORDING 9 + 11 and BREVITY no longer lead every answer with Tesorero.
- ROLE: «el tope solo se atribuye a las empresas que lo traen».

## Bench (sample vs main, `--ignore-latency`)

| Case | main tools / q / s | tick 32 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 5.5 (no invented dueño) |
| alerts | 2 / 6 / 11.0 | 1 / 6 / 7.6 (Tesorero / CFO) |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.6 |
| period-4 | 11 / 2 / 17.0 | 4 / 6 / 9.9 (dos topes + cobros) |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 5.5 (no invented dueño) |

keep=true.

## Decisions

- Same tools. Do not merge tonight.

## Still unknown

- Period lead still sometimes writes bare `0651` / `0030` instead of Empresa. Next tick.
- Loop PID 497061 until 05:55–08:55 +02.
