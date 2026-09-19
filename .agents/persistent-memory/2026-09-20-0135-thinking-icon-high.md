# Thinking chip + effort high

- Author: agent
- Timestamp: 2026-09-20 01:35 +02:00
- Still-binding: Brain on = `reasoning_effort=high` (not `max`). UI shows a **Pensando** chip like a tool row. Do not dump the chain-of-thought.

## What changed

`segmentsInStreamOrder` used to drop `reasoning` parts, so the brain looked idle. It now renders `AgentThinking` (Brain + Pensando + spinner). If the API does not stream reasoning, the chip still appears while the turn is in flight.

Default effort is `high`. Override with `HELMCODE_REASONING_EFFORT`. Bench: `product/web/scripts/bench-thinking.mjs`.
