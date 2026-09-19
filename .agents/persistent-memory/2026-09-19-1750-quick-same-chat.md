# 2026-09-19-1750 — Same Pregunta popup in Rápido

- **Author:** agent (Cursor / Walter)
- **When:** 2026-09-19 17:50 CEST

## What changed

- One Pregunta card on the shared Rápido/Profundo shell (`HealthScoreChat` → `AgentChat` → `/api/ask`).
- Dragging a period on the chart still opens that card and sends the period question. It no longer mounts the old inline `QuickExplain` skeleton.
- Javi’s one-screen Rápido (options ease up, search in the third card) is kept; only the explanation card moved into the popup.
- Same title, chips, tool rows, markdown, and plots. `buildPrompt` stays; it is only the seed text.

## Decisions

- One chat UI and one Ask stack for both modes. The drag-to-explain gesture stays.
- Clearing the period highlight does not close the chat.

## Still unknown

- Local browser check needs `DATABASE_URL`. Confirm on the hosted app after deploy.
