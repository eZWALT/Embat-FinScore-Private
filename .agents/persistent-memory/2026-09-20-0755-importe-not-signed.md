# 2026-09-20 07:55 — reason money field is `importe`

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 07:55 +02
Branch: `agent/overnight-optimize` (tick 77)

## What changed

`slimReason` now returns `importe` (Javi money) instead of `eur`. Prompts (WORDING 7, BREVITY, TOOLS, PRODUCT, ROLE) cite `importe`. UI quotes read `importe` then fall back to `eur`.

## Why

The model treated a field named `eur` as a signed delta (`−535 k€`). The amount is unsigned; the sign lives on `points`.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. `signed_eur` is false on period and why-change. Money still cited (`535 k€`, `34 k€`, `847 €`). Period q=5 from `limitada a 73`. why-change q=4 from that plus the 10 pts/month glide still in the raw `sentence`.

## Decisions

Keep. Do not merge. Next: clip the glide lecture from the model-facing `sentence`, not only the UI quote.

## Still unknown

`72,6` → `73` and method-coda from the unclipped bundle sentence.
