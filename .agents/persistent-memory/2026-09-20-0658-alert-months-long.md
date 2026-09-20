# 2026-09-20 06:58 — Alert months are agosto 2026

Author: cursor-grok-4.6 (overnight Pregunta)
Timestamp: 2026-09-20 06:58 +02

## What changed

`slimAlert.month` uses the long Spanish month (`agosto 2026`), same as `get_company`, so the model does not write `jul 2026`.

## Decisions

- Keep vs main sample suite: 1/6, 1/6, 0/5, 4/6, 1/6. Period 636 chars, still grouped by guard.
- Do not merge to main tonight.

## Still unknown

The model sometimes omits the month on alerts even when the payload has it.
