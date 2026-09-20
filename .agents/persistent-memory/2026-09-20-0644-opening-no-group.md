# 2026-09-20 06:44 — Opening step has no get_group unless asked

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 06:44 +02

## What changed

WORDING 12 is one Spanish/token line. MAP is a short stack index. Sentinel catalog note moved to ROLE. First-step `get_group` is stripped unless the question is about the group / members / pares (`wantsGroupTools`). A period list with «(Grupo 0234)» does not count.

## Decisions

- Keep vs main sample suite: why-score 1/6 (a first slim run called `get_group` in parallel — discarded that opening). alerts 1/6, refuse 0/5, period 4/6, why-change 1/6.
- Helmcode stalled ~96s on why-score; `--ignore-latency`.
- Do not merge to main tonight.

## Still unknown

Period lead sometimes adds a short confidence clause before the bullets. Still better than a coda.
