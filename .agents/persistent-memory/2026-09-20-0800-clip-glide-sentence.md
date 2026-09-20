# 2026-09-20 08:00 — clip glide out of model-facing sentence

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:00 +02
Branch: `agent/overnight-optimize` (tick 78)

## What changed

`clipGlideLecture` lives in `clip-sentence.ts` and runs inside `slimReason`, so the model no longer sees «la puntuación baja como máximo 10 puntos al mes hacia 50 y ahora está limitada a 73». Tool-row quotes use the same helper. Bench `signed_eur` ignores caja/saldo/negativa (a real signed cash figure is not a signed `importe`).

## Why

why-change was reciting the glide lecture from the raw bundle `sentence` (method_coda + `limitada a 73`). UI-only clip was not enough.

## Bench (`--suite sample --ignore-latency`)

Keep vs main. why-change **q=6** (was 4): 72,6 / 81,1 / 535 k€, no glide. Period lead by guard; the only `−1,2 k€` is caja negativa. why-change TTFT stall ~96 s is Helmcode (`--ignore-latency`).

## Decisions

Keep. Do not merge. The bundle sentence still has the lecture in Neon; tools strip it at read time.

## Still unknown

Helmcode ~95 s stalls. Model can still write a signed `importe` on a later draw.
