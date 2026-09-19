# Tools must not kill the answer

- Author: agent
- Timestamp: 2026-09-20 01:48 +02:00
- Still-binding: Cap is still 2 calls per tool per turn (1 for list/plot). After 6 steps or 8 calls, strip tools and write. Do **not** stop the stream on call count — that dropped the text after a parallel burst (e.g. `get_company` ×4 + `get_alerts` ×4). Same-kind row count ticks `×1` → `×n` even when the four parts land in one update. Follow-up chips only after there is answer text.

## What changed

Period questions were finishing on tool chips plus follow-ups and no reply. `stopWhen` treated `totalToolCalls >= 8` as the end of the turn. `prepareStep` now forces a text-only step instead. `AgentTrace` increments the × count on a short interval so a batch does not jump to ×4.
