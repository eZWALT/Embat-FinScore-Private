# 2026-09-20 05:17 — catalog Spanish + parseMonth on explain_change

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 05:17 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `list_companies` catalog matches Empresa / `88,5` / `agosto 2026`.
- `explain_change` accepts `agosto 2026` as well as `YYYY-MM`.

## Decisions

- Fair sample keep (catalog). parseMonth on explain_change is a guard; that tool is rare after the cap.

## Still unknown

- Do not merge. Loop to 05:55–08:55 +02.
