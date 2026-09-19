# 2026-09-19-1808 — Tool wait is a ring, not “Buscando…”

- **Author:** agent (Cursor / Walter)
- **When:** 2026-09-19 18:08 CEST

## What changed

- Running tools and the wait before the first token show a muted Lucide ring (`AgentBusy`). No rotating status phrases.

## Decisions

- Ring over dots or a bar: same icon weight as the wrench. Copy stays off until we ask for it.
