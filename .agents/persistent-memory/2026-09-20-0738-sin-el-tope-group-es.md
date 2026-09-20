# 2026-09-20 07:38 — `sin_el_tope` field; group payload in Spanish

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 07:38 +02

## What changed

`get_company` returns `sin_el_tope` instead of `score_pre_cap`. `get_group` speaks `n_empresas` / `media` / `miembros` / `historial_media`. Prompts cite the new field. Tool-catalog reads both old and new keys.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6. Period and why-change still say «sin el tope sería 81,1». why-score Helmcode stall ~96 s; `--ignore-latency`.
- Do not merge to main tonight.

## Still unknown

Group path is not in the numeric suite.
