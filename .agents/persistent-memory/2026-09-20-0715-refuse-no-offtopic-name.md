# 2026-09-20 07:15 — Recusa without naming the off-topic task

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 07:15 +02

## What changed

WORDING 6 is two sentences (confianza + «sin pagos de facturas»). Tool-catalog summaries read Spanish `grupo_pares` / `metodo` / `origen` / `horizonte_meses` and map `registros no montados`. SCOPE refuse no longer lists «recetas» as echo bait and forbids naming the off-topic task. BREVITY repeats that. Bench extras flag `describes_offtopic` on refuse.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6. Refuse is the 85-char plantilla (`describes_offtopic=false`). First two benches after the WORDING 6 slim echoed «listas de la compra ni recetas» — discarded those as a SCOPE leak.
- Do not merge to main tonight.

## Still unknown

Whether production main refuse still names the paella. Fair compare stays `--suite sample`.
