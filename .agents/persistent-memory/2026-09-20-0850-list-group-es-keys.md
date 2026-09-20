# 2026-09-20 08:50 — list/group member keys in Spanish

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:12 +02
Branch: `agent/overnight-optimize` (tick 85)

## What changed

`list_companies` and `get_group` members: `trayectoria`, `confianza`, `n_alertas`, `gravedad`. Group history points use `media`. Live refuse still the 85-char plantilla, no paella echo.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. **6 / 6 / 5 / 6 / 6**. vs-main +2 +1 +2 +7 +4 (`limite_calque` on main).

## Decisions

Keep. Do not merge. `get_company` still uses `trajectory` as the key (value already Spanish) so PRODUCT/WORDING stay.

## Still unknown

Helmcode stalls. Do not merge tonight.
