# Thinking max + Exa last

- Author: agent
- Timestamp: 2026-09-20 01:20 +02:00
- Still-binding: Brain on = DeepSeek `reasoning_effort=max`. `search_web` (Exa) exists only in Pregunta thinking, after at least one product tool, one call. `EXA_API_KEY` server-only.

## What changed

`llm.ts` sends `max` instead of `high`. `chatTools({ thinking })` adds `search_web`. `prepareStep` hides it until a Neon tool has run. Reply close: **Macroeconomía:** Exa …

## Still unknown

Whether Helmcode forwards `reasoning_effort=max`. Teammate still wiring `EXA_API_KEY` on Vercel.
