# 2026-09-20 08:15 — on-screen chart is a period, not a new plot

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:15 +02
Branch: `agent/overnight-optimize` (tick 80)

## What changed

`wantsPeriodHistory` includes «¿Qué explica este gráfico?». That path no longer loads the plots catalog or offers `plot_series` on the first step. Quote labels use `Empresa N · ` plus a real space before `importe`.

## Why

The chip asks about the chart already on Rápido. Treating it as a plot request loaded unused catalog tokens and put confidence on every `get_company`. Period slim (no confidence, scores-only history) is the same job as a dragged range.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Still **6 / 6 / 5 / 6 / 6**. why-change Helmcode stall ~97 s (`--ignore-latency`).

## Decisions

Keep. Do not merge. `dibuja` / `pinta` / `abanico` still mount plots.

## Still unknown

Live chip after HMR: confirm no confidence coda and spaced quotes.
