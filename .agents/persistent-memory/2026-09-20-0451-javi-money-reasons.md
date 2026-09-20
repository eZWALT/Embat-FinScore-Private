# 2026-09-20 04:51 — Javi money on reason `eur`

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 04:51 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `slimReason` formats `eur` with `formatMoney` so Pregunta cites `34 k€`, not `34.453 €`.
- `.map(slimReason)` must wrap `(reason) => slimReason(reason)` — passing the array index as currency crashed alerts (`1.toUpperCase`).
- `formatMoney` ignores a non-string currency.
- WORDING: cite `eur` as-is.

## Decisions

- Fair sample keep after the map fix. Alerts is the feed (3 alertas, Tesorero / CFO, 32 k€ / 907 k€), not the error confession from the first bench.

| Case | main tools / q / s | tick 14 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 3.3 (705 chars, `34 k€`) |
| alerts | 2 / 6 / 11.0 | 1 / 6 / 6.1 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 0.6 |
| period-4 | 11 / 2 / 17.0 | 4 / 6 / 10.6 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 3.2 |

## Still unknown

- Score still arrives as `88.45`; months still `YYYY-MM`.
