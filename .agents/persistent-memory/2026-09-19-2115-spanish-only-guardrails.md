# Agent replies: Spanish only

- Author: agent
- Timestamp: 2026-09-19 21:15 +02:00
- Still-binding for Pregunta / Sentinel language.

## What changed

SCOPE, ROLE, WORDING, BREVITY, follow-ups and `/api/chat` no longer say «match the user’s language». The user-facing reply (including a refuse) is **only Spanish**. `scope.md` itself is now Spanish so the refuse does not leak English from the layer text.

## Decisions

Technical tokens (`COMP_*`, `kind`, tool names) stay as ids. The sentence around them is Spanish.

## Still unknown

`product_context.md` and `tools_catalog.md` remain English for the model; they are not user copy.
