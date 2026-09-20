# 2026-09-20 08:45 — sentence says «sin el tope», never límite

Author: Cursor overnight (Walter)
Timestamp: 2026-09-20 08:10 +02
Branch: `agent/overnight-optimize` (tick 84)

## What changed

`clipGlideLecture` rewrites «sin el límite / sin la salvaguarda» → «sin el tope sería». WORDING 9: gravity is actuar / vigilar / informativa, never «severidad actuar». Bench extra `limite_calque` (recomputed on both sides; main period/why-change still say salvaguarda).

## Bench (`--suite sample --ignore-latency`)

Keep vs main. Candidate **6 / 6 / 5 / 6 / 6**, no límite. Alerts: «sin el tope sería 58». Period d_quality +7 and why-change +4 vs main because the new extra hits main’s salvaguarda copy.

## Decisions

Keep. Do not merge.

## Still unknown

Helmcode stalls. Bundle JSON still has «sin el límite» in `top_reason`; tools rewrite at read time.
