# 2026-09-20 05:49 — Omit default EUR on get_company

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 05:49 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `get_company` omits `currency` when it is EUR. `eur` already carries `k€`. Other ISO codes still ship.

## Bench

keep=true after a first period-4 that wrote bare `0011 y 0176` (q=5). Second: q=6, Empresa labels, two topes + 0651 mora + 0030 estable.

## Still unknown

- Period lead can still call 0651 «se mueve poco». Do not merge. Min 05:55 / max 08:55 +02.
