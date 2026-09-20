# 2026-09-20 08:55 — get_company keys trayectoria / confianza

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:12 +02
Branch: `agent/overnight-optimize` (tick 86)

## What changed

`get_company` now returns `trayectoria`, `confianza`, `meses_historial`. History rows use `trayectoria`. PRODUCT/ROLE cite those keys. SCOPE fuera-list no longer names cocina (refuse stays the plantilla).

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Still **6 / 6 / 5 / 6 / 6**.

## Decisions

Keep. Do not merge. `confidence_note` stays (bundle note).

## Still unknown

Helmcode stalls. Max window 08:55 +02.
