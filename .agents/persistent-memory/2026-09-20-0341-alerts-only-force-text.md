# 2026-09-20 03:41 — force text after an alerts-only retrieval

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 03:41 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- If the question is only alertas (no índice / por qué / periodo / registros / gráfico), the next step after a successful `get_alerts` is text. Stops the model from thinking about a second tool.

## Decisions

- Fair bench keep. alerts **1 / q 6 / 4.8 s** (TTFT 1.8 s). Tick 8 had a 96 s stall on the same case.
- Production Pregunta on hack-spain still opens the main chips («¿Por qué este índice este mes?», «¿Qué alertas hay?»); this branch is not deployed there.

## Still unknown

- Local FAB can sit under the Next hydration overlay. Visual quote rows need a local Pregunta click when that overlay is gone.
